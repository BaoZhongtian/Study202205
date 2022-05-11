import os
import json
import numpy
import tqdm
from rouge_score import rouge_scorer
import datasets
from transformers import MT5Tokenizer

if __name__ == '__main__':
    metrics = datasets.load_metric('rouge')
    load_path = 'D:/ProjectData/WikiLingualResult/WikiLingualResult-Both2English/'
    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')

    total_label, total_predict = [], []
    for filename in os.listdir(load_path):
        with open(os.path.join(load_path, filename), 'r', encoding='UTF-8') as file:
            sentence = file.readlines()
            if len(sentence) != 2: continue
        total_predict.append(sentence[0])
        total_label.append(sentence[1])
    print(len(total_predict), len(total_label))

    total_score = []
    for index in tqdm.trange(len(total_predict)):
        # metrics.add_batch(
        #     predictions=[
        #         valid_dataset.fields["tgt"].decode(total_predict[index]).replace(' ', '').replace('<unk>', '')],
        #     references=[valid_dataset.fields["tgt"].decode(total_label[index]).replace(' ', '').replace('<unk>', '')])

        result = metrics.add_batch(references=[tokenizer.encode(total_label[index], add_special_tokens=False)],
                                   predictions=[tokenizer.encode(total_predict[index], add_special_tokens=False)])
        # total_score.append([result['rouge1'].fmeasure, result['rouge2'].fmeasure, result['rougeL'].fmeasure])
        # print(total_score)

    result = metrics.compute()
    for sample in result:
        print(sample, result[sample].mid)
    # print(numpy.average(total_score, axis=0))
