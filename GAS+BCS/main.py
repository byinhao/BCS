# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time

from torch.backends import cudnn
from tqdm import tqdm
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import AdamW, T5Tokenizer
from transformers import get_linear_schedule_with_warmup

from data_utils import ABSADataset
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores
from model import MyT5ForConditionalGeneration
import codecs as cs


logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='aste', type=str,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='rest15', type=str,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        default=True,
                        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=5,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--method_name", default='GAS', type=str)

    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--beta", default=0.1, type=float)

    parser.add_argument("--tokenizer", default='../t5_base', type=str)

    args = parser.parse_args()

    # args.model_name_or_path = args.model_name_or_path + f'{args.seed}'

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    output_dir = f"outputs/{args.dataset}_{args.seed}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                       data_type=type_path, max_len=args.max_seq_length,
                       method_name=args.method_name, task=args.task)


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, hparams, tfm_model, tokenizer):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = tfm_model
        self.tokenizer = tokenizer

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        input_length = torch.max((batch["source_mask"] != 0).sum(-1))
        target_length = torch.max((batch["target_mask"] != 0).sum(-1))

        outputs = self(
            input_ids=batch["source_ids"][:, :input_length].contiguous(),
            attention_mask=batch["source_mask"][:, :input_length].contiguous(),
            labels=lm_labels[:, :target_length].contiguous(),
            decoder_attention_mask=batch['target_mask'][:, :target_length].contiguous()
        )
        '''outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )'''

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):

        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(data_loader, model, sents, num_beams):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{args.n_gpu - 1}')
    model.model.to(device)

    model.model.eval()

    outputs, targets = [], []

    for batch in tqdm(data_loader):
        # need to push the data to device
        input_length = torch.max((batch["source_mask"] != 0).sum(-1))
        outs = model.model.generate(input_ids=batch['source_ids'][:, :input_length].contiguous().to(device),
                                    attention_mask=batch['source_mask'][:, :input_length].contiguous().to(device),
                                    max_length=200)  # num_beams=2)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)

    scores, new_results, all_labels, all_preds = compute_scores(outputs, targets, sents, method_name=args.method_name, task=args.task)
    results = {'scores': scores, 'labels': all_labels, 'preds': all_preds}
    # pickle.dump(results, open(f"{args.output_dir}/results-{args.dataset}.pickle", 'wb'))

    return scores, new_results


# initialization
args = init_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# training process
if args.do_train:
    print("\n****** Conduct Training ******")

    # initialize the T5 model
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    tfm_model = MyT5ForConditionalGeneration.from_pretrained(args.model_name_or_path,
                                                             alpha=args.alpha, beta=args.beta,
                                                             task=args.task, method_name=args.method_name)
    model = T5FineTuner(args, tfm_model, tokenizer)

    # prepare for trainer
    train_params = dict(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=[0],
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        callbacks=[LoggingCallback()],
        logger=False,
        checkpoint_callback=False
    )

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    # save the final model
    # model.model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)

    # print("Finish training and saving the model!")

if args.do_eval:
    print("\n****** Conduct inference on trained checkpoint ******")

    # initialize the T5 model from previous checkpoint
    print(f"Load trained model from {args.output_dir}")
    print('Note that a pretrained model is required and `do_true` should be False')
    if not args.do_train:
        # path = '../New_style_unified_framework/outputs/rest15'
        tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        tfm_model = MyT5ForConditionalGeneration.from_pretrained(args.output_dir, alpha=args.alpha, beta=args.beta,
                                                                 task=args.task, method_name=args.method_name)

        model = T5FineTuner(args, tfm_model, tokenizer)

    sents, _ = read_line_examples_from_file(f'data/{args.task}/{args.dataset}/test.txt', silence=False)

    print()
    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset,
                               data_type='test', max_len=args.max_seq_length,
                               method_name=args.method_name, task=args.task)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    # print(test_loader.device)

    # compute the performance scores
    scores, new_results = evaluate(test_loader, model, sents, num_beams=args.num_beams)

    # write to file
    log_file_path = f"results_log/{args.task}/{args.dataset}.txt"
    local_time = time.asctime(time.localtime(time.time()))

    exp_settings = f"Dataset={args.dataset}; seed={args.seed}; alpha={args.alpha}; beta={args.beta}"
    exp_results = f"F1 = {scores['f1']:.4f}\nPrecision = {scores['precision']:.4f}\n" \
                  f"Recall = {scores['recall']:.4f}\nMS = {new_results:.4f}"

    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if not os.path.exists('./results_log'):
        os.mkdir('./results_log')

    if not os.path.exists(f'results_log/{args.task}'):
        os.mkdir(f'results_log/{args.task}')

    with open(log_file_path, "a+") as f:
        f.write(log_str)
