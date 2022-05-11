import os
import torch
import numpy
import datetime
from Tools import ProgressBar
from Loader_WikiLingual import Loader_WikiLingual_Both, SeparateTrainTest
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from ModelStructure.mT5_Based_TripleLingual import MT5_TripleLingual_LateFusion_FinalEmbedding, MT5_TripleLingual_LateFusion_Project

# from Run.Exp_TripleLingual import Treatment
from Run.Exp_TripleLingual_InputLevelFusion import Treatment

cuda_flag = True
if __name__ == '__main__':
    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained(
        'D:/PythonProject/Study202205/mT5_TripleLingual_InputLevelFusion/checkpoint-step-150000')

    if cuda_flag: model = model.cuda()

    train_data, test_data = SeparateTrainTest(Loader_WikiLingual_Both('portuguese', 'spanish'), 'portuguese+spanish')

    save_path = 'WikiLingualResult-Both2English/'
    if not os.path.exists(save_path): os.makedirs(save_path)
    for index, batch in enumerate(test_data):
        print('\rTreating %d' % index, end='')
        if os.path.exists(save_path + '%08d.csv' % index): continue
        with open(save_path + '%08d.csv' % index, 'w') as file:
            pass
        input_ids, label_ids = Treatment(batch)

        result = model.generate(input_ids, min_length=int(0.75 * len(label_ids[0])),
                                max_length=int(1.5 * len(label_ids[0])),
                                num_beams=16, repetition_penalty=5.0).detach().cpu().numpy()
        with open(save_path + '%08d.csv' % index, 'w', encoding='UTF-8') as file:
            file.write(tokenizer.batch_decode(result, skip_special_tokens=True)[0])
            file.write('\n')
            file.write(batch['EnglishSummary'])
