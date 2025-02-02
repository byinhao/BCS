# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset
from itertools import permutations

sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}


def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            line = line.lower()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels


def get_target_sequence(method_name, task, tuples):
    if task == 'asqp':
        if method_name == 'GAS':
            target = []
            for tuple in tuples:
                one = f'( {tuple[0]}, {tuple[1]}, {tuple[2]}, {tuple[3]} )'
                target.append(one)
            target = ' ; '.join(target)
            return target
        elif method_name == 'Special_Symbols':
            target = []
            for tuple in tuples:
                at, ac, sp, ot = tuple
                man_ot = sentword2opinion[sp]
                if at == 'null':
                    at = 'it'
                one = f'[AT] {at} [OT] {ot} [AC] {ac} [SP] {man_ot}'
                target.append(one)
            target = ' [SSEP] '.join(target)
            return target
        elif method_name == 'Paraphrase':
            target = []
            for tuple in tuples:
                at, ac, sp, ot = tuple
                man_ot = sentword2opinion[sp]
                if at == 'null':
                    at = 'it'
                one = f'{ac} is {man_ot} because {at} is {ot}'
                target.append(one)
            target = ' [SSEP] '.join(target)
            return target
    elif task == 'aste':
        if method_name == 'GAS':
            target = []
            for tuple in tuples:
                one = f'( {tuple[0]}, {tuple[1]}, {tuple[2]} )'
                target.append(one)
            target = ' ; '.join(target)
            return target
    elif task == 'tasd':
        if method_name == 'GAS':
            target = []
            for tuple in tuples:
                one = f'( {tuple[0]}, {tuple[1]}, {tuple[2]} )'
                target.append(one)
            target = ' ; '.join(target)
            return target
    elif task == 'rte':
        if method_name == 'GAS':
            target = []
            for tuple in tuples:
                one = f'( {tuple[0]}, {tuple[1]}, {tuple[2]} )'
                target.append(one)
            target = ' ; '.join(target)
            return target
    elif task == 'rqe':
        if method_name == 'GAS':
            target = []
            for tuple in tuples:
                one = f'( {tuple[0]}, {tuple[1]}, {tuple[2]}, {tuple[3]}, {tuple[4]} )'
                target.append(one)
            target = ' ; '.join(target)
            return target
    elif task == 'ner':
        if method_name == 'GAS':
            target = []
            for tuple in tuples:
                one = f'( {tuple[0]}, {tuple[1]} )'
                target.append(one)
            target = ' ; '.join(target)
            return target
    else:
        raise NotImplementedError

def get_inputs_targets(sents, labels, data_path, method_name, task):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    inputs = []
    for i in range(len(labels)):
        cur_sent = sents[i]
        cur_sent = ' '.join(cur_sent)
        cur_inputs = cur_sent
        label = labels[i]
        target = get_target_sequence(method_name, task, label)
        inputs.append(cur_inputs)
        targets.append(target)
    return inputs, targets


def get_transformed_io(data_path, data_dir, method_name, task):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path, silence=False)

    inputs, targets = get_inputs_targets(sents, labels, data_path, method_name, task)

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, method_name, task, max_len=128):
        self.data_path = f'data/{task}/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.data_type = data_type
        self.method_name = method_name
        self.task = task
        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "labels": self.all_labels[index]}

    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path, self.data_dir, self.method_name, self.task)
        self.all_labels = targets
        for i in range(len(inputs)):
            # change input and target to two strings
            # input = ' '.join(inputs[i])
            input = inputs[i]
            target = targets[i]
            tokenized_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            if self.data_type == 'train':
                tokenized_target = self.tokenizer.batch_encode_plus(
                    [target], max_length=self.max_len, padding="max_length",
                    truncation=True, return_tensors="pt"
                )
            else:
                tokenized_target = self.tokenizer.batch_encode_plus(
                    [target], max_length=1024, padding="max_length",
                    truncation=True, return_tensors="pt"
                )
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
