#
#  xlsx2bleu.py
#     > python csv2bleu.py xxx.csv ref_name pred_name
#

from datasets import load_dataset, load_metric, Dataset
import pandas as pd
import sys
from collections import defaultdict
import json
#from sacrebleu.metrics import BLEU, CHRF
# for rouge evaluation, use unicode in place of chinese
def convert_chinese_to_unicode(sentence):
    char_list = []
    for char in sentence:
        code = ord(char)
        if code < 256:
            char_list.append(char)
        else:
            char_list.append(f' {code:x} ')

    return ''.join(char_list)

def rouge_metric_test():
    #gao =  s.decode(encoding='utf-8')
    gao = '高'
    gao_code = ord(gao)
    print(f'{gao_code:x}')

    predictions = ["hello there 高元", "general kenobi"]
    references = ["hello there 高", "general kenobi"]
    predictions_unicode = [convert_chinese_to_unicode(pred) for pred in predictions]
    references_unicode = [convert_chinese_to_unicode(ref) for ref in references]

    rouge_metric = load_metric("rouge")
    results = rouge_metric.compute(predictions=predictions_unicode, references=references_unicode)
    print(results)

# ========
def evaluate_bleu_rouge(excel_fname, ref_col_name, pred_col_name):
    # csv_fname = 'web_bai_eval_74.csv'
    df = pd.read_excel(excel_fname)
    rows = df.shape[0]
    cols = df.shape[1]
    col_names = ' ,'.join(list(df.columns))
    print(f'\'{excel_fname}\' read with {rows} rows, {cols} columns.')
    print(f'{cols} columns are [{col_names}]')

    prediction_list = df[pred_col_name].to_list()
    reference_list = df[ref_col_name].to_list()

    prediction_token_list = [' '.join(list(pred)) for pred in prediction_list ]
    reference_token_list = [' '.join(list(ref)) for ref in reference_list ]
    reference_token_list2 = [[ref] for ref in reference_token_list]

    prediction_token_unicode_list = [convert_chinese_to_unicode(pred) for pred in prediction_token_list]
    reference_token_unicode_list = [convert_chinese_to_unicode(ref) for ref in reference_token_list]

    bleu_metric = load_metric("sacrebleu")
    # bleu_metric.add(
    #    prediction="the the the the the the", reference=["the cat is on the mat"])
    # results = bleu_metric.compute(smooth_method="floor", smooth_value=0)
   # chrf_metric = load_metric("chrf")

    rouge_metric = load_metric("rouge")
    # rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    #chrf_metric.add_batch(predictions=prediction_token_list, references=reference_token_list2)
    bleu_metric.add_batch(predictions=prediction_token_list, references=reference_token_list2)
    rouge_metric.add_batch(predictions=prediction_token_unicode_list, references=reference_token_unicode_list)

    #chrf_result = chrf_metric.compute()
    bleu_result = bleu_metric.compute()
    rouge_result = rouge_metric.compute()

    score_dict={'bleu':{},'rouge':{}}
    """print(f'\nchrf =')
    for key in chrf_result:
        print(f'\t{key}: {chrf_result[key]}')
        score_dict['chrf'].update({key:chrf_result[key]})"""
        
    print(f'\nbleu =')
    for key in bleu_result:
        print(f'\t{key}: {bleu_result[key]}')
        score_dict['bleu'].update({key:bleu_result[key]})
        #[key]={key:bleu_result[key]}
        
    
    print(f'\nrouge =')
    #score_list.append('rouge')
    for key in rouge_result:
        rouge = rouge_result[key]
        low = rouge.low
        mid =rouge.mid
        high =rouge.high
        

        print(f'\t{key}:\tp = [{low.precision:.2f}, {mid.precision:.2f}, {high.precision:.2f}],', end='')
        print(f'\tr = [{low.recall:.2f}, {mid.recall:.2f}, {high.recall:.2f}],', end='')
        print(f'\tf1 = [{low.fmeasure:.2f}, {mid.fmeasure:.2f}, {high.fmeasure:.2f}]')
        score_dict['rouge'].update({key:{'p':[low.precision,mid.precision,high.precision],
                                 'r':[low.recall,mid.recall,high.recall],
                                 'f1':[low.fmeasure,mid.fmeasure,high.fmeasure]}})
        
        
    #print(score_dict)
    
   # with open("json_best/raynardj_b2w_mytf_nb8_nore2_train_5480.json", 'w') as fp:
       # json.dump(score_dict, fp)
    

if __name__ == '__main__':
    csv_fname = 'raynardi_e0_test_21923.csv'
    ref_col_name = 'baihua'
    pred_col_name = 'predict_baihua'

    if len(sys.argv) == 4:
        args = sys.argv[1:]
        csv_fname = args[0]
        ref_col_name = args[1]
        pred_col_name = args[2]

    print(f'usage: python csv2bleu.py {csv_fname} {ref_col_name} {pred_col_name}')
    evaluate_bleu_rouge(csv_fname, ref_col_name, pred_col_name)
