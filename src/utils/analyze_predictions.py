import json
import numpy as np
def compute_mean_and_deviation(data):
    data = np.array(data)
    mean = np.mean(data)
    dev = np.sqrt(np.mean((data-mean)**2))
    mean = round(mean,3)
    dev = round(dev,3)
    return mean,dev
def analyze_predictions(predictions, description=''):
    # template of a prediction
    # { 'Task': 'RE', 'Dataset': 'ADE_corpus', 
    #   'Instance': {'id': '0', 'sentence': 'xxx', 
    #   'label': '<extra_id_0> <extra_id_0>  <extra_id_5> recall pneumonitis <extra_id_0> adverse effect <extra_id_5> Gemcitabine <extra_id_1> <extra_id_1> <extra_id_1>', 
    #   'instruction': '<asoc> adverse effect<extra_id_2> {0}', 
    #   'ground_truth': [['recall pneumonitis', 'adverse effect', 'Gemcitabine']], 
    #   'answer_prefix': 'Answer:'}, 
    #   'Prediction': [['recall pneumonitis', 'adverse effect', 'Gemcitabine']]}
    pred_nums = []
    dup_nums = []
    for pred in predictions:
        pred_num = len(pred['Prediction'])
        dup_num = pred_num - len(set(['/'.join(t) for t in pred['Prediction']]))
        pred_nums.append(pred_num)
        dup_nums.append(dup_num)
    mean_pred_num, dev_pred_num = compute_mean_and_deviation(pred_nums)
    mean_dup_num, dev_dup_num = compute_mean_and_deviation(dup_nums)
    print('Prediction num for {}: {} +- {}'.format(description, mean_pred_num, dev_pred_num))
    print('Duplication num for {}: {} +- {}'.format(description, mean_dup_num, dev_dup_num))
def compute_f1_category(predictions):
    tp = {}
    fp = {}
    fn = {}
    precisions = {}
    recalls = {}
    f1s = {}
    for pred in predictions:
        name = pred['Task'] + '_' + pred['Dataset']
        if name not in tp:
            tp[name] = {}
            fp[name] = {}
            fn[name] = {}
            tp[name]['all'] = 0
            fp[name]['all'] = 0
            fn[name]['all'] = 0
        pred_set = set(['/!@'.join(t) for t in pred['Prediction']])
        gold_set = set(['/!@'.join(t) for t in pred['Instance']['ground_truth']])
        tp[name]['all'] += len(pred_set & gold_set)
        fp[name]['all'] += len(pred_set - gold_set)
        fn[name]['all'] += len(gold_set - pred_set)
        for t in gold_set:
            category = t.split('/!@')[1]
            assert category != 'all'
            if category not in tp[name]:
                tp[name][category] = 0
                fp[name][category] = 0
                fn[name][category] = 0
            if t in pred_set:
                tp[name][category] += 1
            else:
                fn[name][category] += 1
        for t in pred_set:
            category = t.split('/!@')[1]
            assert category != 'all'
            if category not in tp[name]:
                tp[name][category] = 0
                fp[name][category] = 0
                fn[name][category] = 0
            if t not in gold_set:
                fp[name][category] += 1
    for name in tp:
        precisions[name] = {}
        recalls[name] = {}
        f1s[name] = {}
        for category in tp[name]:
            tp_num = tp[name][category]
            fp_num = fp[name][category]
            fn_num = fn[name][category]
            precision = tp_num/(tp_num+fp_num) if tp_num+fp_num > 0 else 0
            recall = tp_num/(tp_num+fn_num) if tp_num+fn_num > 0 else 0
            f1 = 2*precision*recall/(precision+recall) if precision+recall > 0 else 0
            precisions[name][category] = precision
            recalls[name][category] = recall
            f1s[name][category] = f1
    return precisions, recalls, f1s
if __name__ == '__main__':
    path = '/storage/zkhu/UIE-pp/output/re/ADE_corpus_ddi13/uie_fft/best_model_for_ddi13/predict_eval_predictions.jsonl'
    with open(path) as fin:
        lines = fin.readlines()
        test_predictions = [json.loads(l) for l in lines]

    precisions, recalls, f1s = compute_f1_category(test_predictions)
    # output 
    from tabulate import tabulate

    table = []
    headers = ['Name', 'Category', 'Precision', 'Recall', 'F1 Score']

    for name in precisions:
        for category in precisions[name]:
            precision = round(precisions[name][category], 4)
            recall = round(recalls[name][category], 4)
            f1 = round(f1s[name][category], 4)
            table.append([name, category, precision, recall, f1])

    print(tabulate(table, headers, tablefmt='grid'))

    
    