# -*- coding: utf-8 -*-

import numpy as np

# This script handles the decoding functions and performance measurement

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}


def extract_spans_para(task, method_name, seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split(';')]
    if task == 'asqp':
        if method_name == 'GAS':
            for s in sents:
                # food quality is bad because pizza is over cooked.
                try:
                    at, ac, sp, ot = s.split(',')
                    at = at.strip()
                    ac = ac.strip()
                    sp = sp.strip()
                    ot = ot.strip()
                    if len(at) >= 1 and at[0] == '(':
                        at = at[1:].strip()
                    if len(ot) >= 1 and ot[-1] == ')':
                        ot = ot[:-1].strip()
                except ValueError:
                    try:
                        # print(f'In {seq_type} seq, cannot decode: {s}')
                        pass
                    except UnicodeEncodeError:
                        # print(f'In {seq_type} seq, a string cannot be decoded')
                        pass
                    ac, at, sp, ot = '', '', '', ''

                quads.append((ac, at, sp, ot))
    elif task == 'aste':
        if method_name == 'GAS':
            for s in sents:
                # food quality is bad because pizza is over cooked.
                try:
                    at, ot, sp = s.split(',')
                    at = at.strip()
                    ot = ot.strip()
                    sp = sp.strip()
                    if len(at) >= 1 and at[0] == '(':
                        at = at[1:].strip()
                    if len(sp) >= 1 and sp[-1] == ')':
                        sp = sp[:-1].strip()
                except ValueError:
                    try:
                        # print(f'In {seq_type} seq, cannot decode: {s}')
                        pass
                    except UnicodeEncodeError:
                        # print(f'In {seq_type} seq, a string cannot be decoded')
                        pass
                    at, ot, sp = '', '', ''

                quads.append((at, ot, sp))
    elif task == 'tasd':
        if method_name == 'GAS':
            for s in sents:
                # food quality is bad because pizza is over cooked.
                try:
                    at, ac, sp = s.split(',')
                    at = at.strip()
                    ac = ac.strip()
                    sp = sp.strip()
                    if len(at) >= 1 and at[0] == '(':
                        at = at[1:].strip()
                    if len(sp) >= 1 and sp[-1] == ')':
                        sp = sp[:-1].strip()
                except ValueError:
                    try:
                        # print(f'In {seq_type} seq, cannot decode: {s}')
                        pass
                    except UnicodeEncodeError:
                        # print(f'In {seq_type} seq, a string cannot be decoded')
                        pass
                    at, ac, sp = '', '', ''

                quads.append((at, ac, sp))
    elif task == 'rte':
        if method_name == 'GAS':
            for s in sents:
                # food quality is bad because pizza is over cooked.
                try:
                    ele1, ele2, ele3 = s.split(',')
                    ele1 = ele1.strip()
                    ele2 = ele2.strip()
                    ele3 = ele3.strip()
                    if len(ele1) >= 1 and ele1[0] == '(':
                        ele1 = ele1[1:].strip()
                    if len(ele3) >= 1 and ele3[-1] == ')':
                        ele3 = ele3[:-1].strip()
                except ValueError:
                    try:
                        # print(f'In {seq_type} seq, cannot decode: {s}')
                        pass
                    except UnicodeEncodeError:
                        # print(f'In {seq_type} seq, a string cannot be decoded')
                        pass
                    ele1, ele2, ele3 = '', '', ''

                quads.append((ele1, ele2, ele3))
    elif task == 'rqe':
        if method_name == 'GAS':
            for s in sents:
                # food quality is bad because pizza is over cooked.
                try:
                    ele1, ele2, ele3, ele4, ele5 = s.split(',')
                    ele1 = ele1.strip()
                    ele2 = ele2.strip()
                    ele3 = ele3.strip()
                    ele4 = ele4.strip()
                    ele5 = ele5.strip()
                    if len(ele1) >= 1 and ele1[0] == '(':
                        ele1 = ele1[1:].strip()
                    if len(ele5) >= 1 and ele5[-1] == ')':
                        ele5 = ele5[:-1].strip()
                except ValueError:
                    try:
                        # print(f'In {seq_type} seq, cannot decode: {s}')
                        pass
                    except UnicodeEncodeError:
                        # print(f'In {seq_type} seq, a string cannot be decoded')
                        pass
                    ele1, ele2, ele3, ele4, ele5 = '', '', '', '', ''

                quads.append((ele1, ele2, ele3, ele4, ele5))
    elif task == 'ner':
        if method_name == 'GAS':
            for s in sents:
                # food quality is bad because pizza is over cooked.
                try:
                    ele1, ele2 = s.split(',')
                    ele1 = ele1.strip()
                    ele2 = ele2.strip()
                    if len(ele1) >= 1 and ele1[0] == '(':
                        ele1 = ele1[1:].strip()
                    if len(ele2) >= 1 and ele2[-1] == ')':
                        ele2 = ele2[:-1].strip()
                except ValueError:
                    try:
                        # print(f'In {seq_type} seq, cannot decode: {s}')
                        pass
                    except UnicodeEncodeError:
                        # print(f'In {seq_type} seq, a string cannot be decoded')
                        pass
                    ele1, ele2 = '', ''

                quads.append((ele1, ele2))
    else:
        raise NotImplementedError
    return quads


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_new_scores(all_preds, all_golds):
    all_final_scores = []
    n_ele = len(all_golds[0][0])
    one_score = 1 / n_ele
    for i in range(len(all_golds)):
        preds = all_preds[i]
        golds = all_golds[i]
        preds_sets = []
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
        if len(preds) == 0 and len(golds) == 0:
            final_score = 1
        elif len(preds) == 0 and len(golds) != 0:
            final_score = 0
        elif len(preds) != 0 and len(golds) == 0:
            final_score = 0
        else:
            final_score = (n_intersection / n_union) * sum(all_scores) / len(preds)
        all_final_scores.append(final_score)
    new_scores = sum(all_final_scores) / len(all_final_scores)
    return new_scores


def compute_scores(pred_seqs, gold_seqs, sents, method_name, task):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        if gold_seqs[i] != '':
            gold_list = extract_spans_para(task, method_name, gold_seqs[i], 'gold')
        else:
            gold_list = []
        if pred_seqs[i] != '':
            pred_list = extract_spans_para(task, method_name, pred_seqs[i], 'pred')
        else:
            pred_list = []

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels)
    print(scores)

    print("\nNew results:")
    new_results = compute_new_scores(all_preds, all_labels)
    print(new_results)

    return scores, new_results, all_labels, all_preds
