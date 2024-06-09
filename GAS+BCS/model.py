import numpy as np
import torch.nn as nn
import torch
#from transformers.modeling_t5 import *
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import *
from transformers.file_utils import ModelOutput
from transformers.generation_utils import *
from transformers.generation_beam_search import *
import copy

_CONFIG_FOR_DOC = "T5Config"

PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.
    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.
    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:
                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24
    Example:
    ```python
    # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained('t5-3b')
    device_map = {0: [0, 1, 2],
             1: [3, 4, 5, 6, 7, 8, 9],
             2: [10, 11, 12, 13, 14, 15, 16],
             3: [17, 18, 19, 20, 21, 22, 23]}
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.
    Example:
    ```python
    # On a 4 GPU machine with t5-3b:
    model = T5ForConditionalGeneration.from_pretrained('t5-3b')
    device_map = {0: [0, 1, 2],
                 1: [3, 4, 5, 6, 7, 8, 9],
                 2: [10, 11, 12, 13, 14, 15, 16],
                 3: [17, 18, 19, 20, 21, 22, 23]}
    model.parallelize(device_map) # Splits the model across several devices
    model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""

add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class MyT5ForConditionalGeneration(T5PreTrainedModel):
    authorized_missing_keys = [r"encoder\.embed_tokens\.weight", r"decoder\.embed_tokens\.weight", r"lm_head\.weight"]

    def __init__(self, config, alpha, beta, task, method_name):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.alpha = alpha
        self.beta = beta
        self.task = task
        self.method_name = method_name

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def compute_kl_loss(self, p, q, decoder_attention_mask, pad_mask=None):

        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        # gold_num = (decoder_attention_mask != 0).sum()
        p_loss = (p_loss * decoder_attention_mask).sum() # / gold_num
        q_loss = (q_loss * decoder_attention_mask).sum() # / gold_num

        loss = (p_loss + q_loss) / 2
        return loss

    def compute_dirichlet_uncertainty(self, lm_logits):
        softplus = nn.Softplus()
        evidence_logits = softplus(lm_logits)
        uncertainty = lm_logits.shape[-1] / (evidence_logits + 1).sum(-1)
        return uncertainty

    def get_choosed(self, probability, length):
        # p = torch.pow((1 - probability), length.unsqueeze(-1))
        # mask = torch.bernoulli(1 - p).long()

        # Construct binomial distributions
        binomial_distribution = torch.distributions.Binomial(total_count=length, probs=probability)

        # Sample each binomial distribution
        samples = binomial_distribution.sample()

        # Once succeed, it can be selected.
        choosed = torch.where(samples > 0, 1, 0)
        return choosed

    def _strip(self, ele):
        if len(ele) >= 1 and ele[0] == '3':
            ele = ele[1:]
        if len(ele) >= 1 and ele[-1] == '3':
            ele = ele[:-1]
        return ele

    def get_lists(self, ele):
        return ele.strip().split()

    def get_tuples(self, seq, dtype):
        if self.task == 'asqp':
            if self.method_name == 'GAS':
                tuples = []
                if dtype == 'pred':
                    seq = seq[: (seq != 0).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                else:
                    seq = seq[: (seq != -100).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                seq = ' '.join([f"{w}" for w in seq.tolist()])
                sents = [s for s in seq.split(' 117 ')] # 117 -> ;
                for s in sents:
                    try:
                        at, ac, sp, ot = s.split(' 6 ') # 6 -> ,
                        at = at.strip().split()
                        ac = ac.strip().split()
                        sp = sp.strip().split()
                        ot = ot.strip().split()
                        at = self._strip(at)
                        ac = self._strip(ac)
                        sp = self._strip(sp)
                        ot = self._strip(ot)
                        if len(at) >= 1 and at[0] == '41':
                            at = self._strip(at[1:])
                        if len(ot) >= 1 and ot[-1] == '61':
                            ot = self._strip(ot[:-1])
                        at, ac, sp, ot = ' '.join(at), ' '.join(ac), ' '.join(sp), ' '.join(ot)
                    except ValueError:
                        try:
                            # print(f'In {seq_type} seq, cannot decode: {s}')
                            pass
                        except UnicodeEncodeError:
                            # print(f'In {seq_type} seq, a string cannot be decoded')
                            pass
                        ac, at, sp, ot = '', '', '', ''

                    tuples.append((ac, at, sp, ot))
        elif self.task == 'aste':
            if self.method_name == 'GAS':
                tuples = []
                if dtype == 'pred':
                    seq = seq[: (seq != 0).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                else:
                    seq = seq[: (seq != -100).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                seq = ' '.join([f"{w}" for w in seq.tolist()])
                sents = [s for s in seq.split(' 117 ')]  # 117 -> ;
                for s in sents:
                    try:
                        at, ot, sp = s.split(' 6 ')  # 6 -> ,
                        at = at.strip().split()
                        ot = ot.strip().split()
                        sp = sp.strip().split()
                        at = self._strip(at)
                        ot = self._strip(ot)
                        sp = self._strip(sp)
                        if len(at) >= 1 and at[0] == '41':
                            at = self._strip(at[1:])
                        if len(sp) >= 1 and sp[-1] == '61':
                            sp = self._strip(sp[:-1])
                        at, ot, sp = ' '.join(at), ' '.join(ot), ' '.join(sp)
                    except ValueError:
                        try:
                            # print(f'In {seq_type} seq, cannot decode: {s}')
                            pass
                        except UnicodeEncodeError:
                            # print(f'In {seq_type} seq, a string cannot be decoded')
                            pass
                        at, ot, sp = '', '', ''

                    tuples.append((at, ot, sp))
        elif self.task == 'tasd':
            if self.method_name == 'GAS':
                tuples = []
                if dtype == 'pred':
                    seq = seq[: (seq != 0).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                else:
                    seq = seq[: (seq != -100).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                seq = ' '.join([f"{w}" for w in seq.tolist()])
                sents = [s for s in seq.split(' 117 ')]  # 117 -> ;
                for s in sents:
                    try:
                        at, ac, sp = s.split(' 6 ')  # 6 -> ,
                        at = at.strip().split()
                        ac = ac.strip().split()
                        sp = sp.strip().split()
                        at = self._strip(at)
                        ac = self._strip(ac)
                        sp = self._strip(sp)
                        if len(at) >= 1 and at[0] == '41':
                            at = self._strip(at[1:])
                        if len(sp) >= 1 and sp[-1] == '61':
                            sp = self._strip(sp[:-1])
                        at, ac, sp = ' '.join(at), ' '.join(ac), ' '.join(sp)
                    except ValueError:
                        try:
                            # print(f'In {seq_type} seq, cannot decode: {s}')
                            pass
                        except UnicodeEncodeError:
                            # print(f'In {seq_type} seq, a string cannot be decoded')
                            pass
                        at, ac, sp = '', '', ''

                    tuples.append((at, ac, sp))
        elif self.task == 'rte':
            if self.method_name == 'GAS':
                tuples = []
                if dtype == 'pred':
                    seq = seq[: (seq != 0).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                else:
                    seq = seq[: (seq != -100).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                seq = ' '.join([f"{w}" for w in seq.tolist()])
                sents = [s for s in seq.split(' 117 ')]  # 117 -> ;
                for s in sents:
                    try:
                        ele1, ele2, ele3 = s.split(' 6 ')  # 6 -> ,
                        ele1 = ele1.strip().split()
                        ele2 = ele2.strip().split()
                        ele3 = ele3.strip().split()
                        ele1 = self._strip(ele1)
                        ele2 = self._strip(ele2)
                        ele3 = self._strip(ele3)
                        if len(ele1) >= 1 and ele1[0] == '41':
                            ele1 = self._strip(ele1[1:])
                        if len(ele3) >= 1 and ele3[-1] == '61':
                            ele3 = self._strip(ele3[:-1])
                        ele1, ele2, ele3 = ' '.join(ele1), ' '.join(ele2), ' '.join(ele3)
                    except ValueError:
                        try:
                            # print(f'In {seq_type} seq, cannot decode: {s}')
                            pass
                        except UnicodeEncodeError:
                            # print(f'In {seq_type} seq, a string cannot be decoded')
                            pass
                        ele1, ele2, ele3 = '', '', ''

                    tuples.append((ele1, ele2, ele3))
        elif self.task == 'rqe':
            if self.method_name == 'GAS':
                tuples = []
                if dtype == 'pred':
                    seq = seq[: (seq != 0).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                else:
                    seq = seq[: (seq != -100).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                seq = ' '.join([f"{w}" for w in seq.tolist()])
                sents = [s for s in seq.split(' 117 ')]  # 117 -> ;
                for s in sents:
                    try:
                        ele1, ele2, ele3, ele4, ele5 = s.split(' 6 ')  # 6 -> ,
                        ele1 = ele1.strip().split()
                        ele2 = ele2.strip().split()
                        ele3 = ele3.strip().split()
                        ele4 = ele4.strip().split()
                        ele5 = ele5.strip().split()
                        ele1 = self._strip(ele1)
                        ele2 = self._strip(ele2)
                        ele3 = self._strip(ele3)
                        ele4 = self._strip(ele4)
                        ele5 = self._strip(ele5)
                        if len(ele1) >= 1 and ele1[0] == '41':
                            ele1 = self._strip(ele1[1:])
                        if len(ele5) >= 1 and ele5[-1] == '61':
                            ele5 = self._strip(ele5[:-1])
                        ele1, ele2, ele3, ele4, ele5 = ' '.join(ele1), ' '.join(ele2), ' '.join(ele3), ' '.join(ele4), ' '.join(ele5)
                    except ValueError:
                        try:
                            # print(f'In {seq_type} seq, cannot decode: {s}')
                            pass
                        except UnicodeEncodeError:
                            # print(f'In {seq_type} seq, a string cannot be decoded')
                            pass
                        ele1, ele2, ele3, ele4, ele5 = '', '', '', '', ''

                    tuples.append((ele1, ele2, ele3, ele4, ele5))
        elif self.task == 'ner':
            if self.method_name == 'GAS':
                tuples = []
                if dtype == 'pred':
                    seq = seq[: (seq != 0).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                else:
                    seq = seq[: (seq != -100).sum()]
                    if seq[-1] == 1:
                        seq = seq[:-1]
                seq = ' '.join([f"{w}" for w in seq.tolist()])
                sents = [s for s in seq.split(' 117 ')]  # 117 -> ;
                for s in sents:
                    try:
                        ele1, ele2 = s.split(' 6 ')  # 6 -> ,
                        ele1 = ele1.strip().split()
                        ele2 = ele2.strip().split()
                        ele1 = self._strip(ele1)
                        ele2 = self._strip(ele2)
                        if len(ele1) >= 1 and ele1[0] == '41':
                            ele1 = self._strip(ele1[1:])
                        if len(ele2) >= 1 and ele2[-1] == '61':
                            ele2 = self._strip(ele2[:-1])
                        ele1, ele2 = ' '.join(ele1), ' '.join(ele2)
                    except ValueError:
                        try:
                            # print(f'In {seq_type} seq, cannot decode: {s}')
                            pass
                        except UnicodeEncodeError:
                            # print(f'In {seq_type} seq, a string cannot be decoded')
                            pass
                        ele1, ele2 = '', ''

                    tuples.append((ele1, ele2))
        else:
            raise NotImplementedError
        return tuples

    def compute_matching_score(self, preds, golds):
        preds_sets = []
        n_ele = len(golds[0])
        one_score = 1 / n_ele
        for quad in preds:
            for element in quad:
                if element not in preds_sets:
                    preds_sets.append(element)
        golds_sets = []
        for quad in golds:
            for element in quad:
                if element not in golds_sets:
                    golds_sets.append(element)
        n_intersection = 0.0
        n_union = 0.0
        for ele in preds_sets:
            if ele in golds_sets:
                n_intersection += 1.0
        temp_sets = []
        for ele in golds_sets + preds_sets:
            if ele not in temp_sets:
                temp_sets.append(ele)
                n_union += 1.0
        all_scores = []
        for pred in preds:
            cur_max_score = 0.0
            for gold in golds:
                n = 0.0
                for i in range(n_ele):
                    if pred[i] == gold[i]:
                        n += 1
                if cur_max_score < n * one_score:
                    cur_max_score = n * one_score
            all_scores.append(cur_max_score)
        final_score = (n_intersection / n_union) * sum(all_scores) / len(preds)
        return final_score

    def get_scores(self, preds, golds):
        scores = torch.zeros(golds.shape[0], device=golds.device, dtype=torch.float32)
        for i in range(len(golds)):
            pred = preds[i]
            gold = golds[i]
            # According to the corresponding rules, convert the sequence into a tuple.
            tuple_preds = self.get_tuples(pred, 'pred')
            tuple_golds = self.get_tuples(gold, 'gold')

            # compute Matching Score
            score = self.compute_matching_score(tuple_preds, tuple_golds)
            scores[i] = score
        return scores

    def uncertain_normalization(self, uncertainty, decoder_attention_mask):
        cur_max = torch.max(uncertainty * decoder_attention_mask, dim=-1).values
        decoder_attention_mask[decoder_attention_mask == 0] = 100
        cur_min = torch.min(uncertainty * decoder_attention_mask, dim=-1).values
        diff = torch.clamp((cur_max - cur_min), min=0.0001)
        probability = torch.clamp(torch.div(cur_max.unsqueeze(-1) - uncertainty, diff.unsqueeze(-1)), min=0.0001,
                                  max=0.9999)
        return probability

    def obtain_counterpart_sequence(self, lm_logits, labels, decoder_attention_mask):
        # local uncertainty
        local_uncertainty = self.compute_dirichlet_uncertainty(lm_logits)

        # normalization
        probability = self.uncertain_normalization(local_uncertainty, decoder_attention_mask)

        # predicted sequence
        preds = torch.argmax(lm_logits, dim=-1) * decoder_attention_mask
        # All Matching Score
        score = self.get_scores(preds, labels)
        length = (labels != -100).sum(-1)

        choosed = self.get_choosed(probability, (1 - score) * length)
        counterpart_sequence = labels * (1 - choosed) + preds * choosed

        return counterpart_sequence

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """

        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            with torch.no_grad():
                counterpart_sequence = self.obtain_counterpart_sequence(lm_logits, labels, decoder_attention_mask)

                decoder_counterpart_sequence = self._shift_right(counterpart_sequence)
            # Decode 1
            decoder_outputs1 = self.decoder(
                input_ids=decoder_counterpart_sequence,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output1 = decoder_outputs1[0]
            sequence_output1 = sequence_output1 * (self.model_dim ** -0.5)
            lm_logits1 = self.lm_head(sequence_output1)

            # Semantic Alignment Loss
            kl_loss = self.compute_kl_loss(lm_logits, lm_logits1, decoder_attention_mask.unsqueeze(-1))

            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss1 = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss2 = loss_fct(lm_logits1.view(-1, lm_logits.size(-1)), labels.view(-1))

            loss = (1 - self.alpha) * loss1 + self.alpha * loss2 + self.beta * kl_loss

            #lm_logits_max, lm_logits_max_index = torch.max(lm_logits, dim=-1)
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
