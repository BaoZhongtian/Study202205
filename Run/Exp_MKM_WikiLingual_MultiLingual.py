import os
import torch
import numpy
import datetime
from Tools import ProgressBar
from MaskedKeywordsTreatment import MaskedKeywordsModule
from Loader_WikiLingual import Loader_WikiLingual, SeparateTrainTest
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

cuda_flag = True


def treatment(batch):
    treat_lingual = batch['CrossLingualName']
    treat_cls_article, treat_cls_summary = batch['CrossLingualDocument'], batch['CrossLingualSummary']
    masked_result = None
    if treat_lingual == 'portuguese': masked_result = \
        portuguese_masked.mask_overlap(treat_cls_article, treat_cls_summary)
    if treat_lingual == 'spanish': masked_result = \
        spanish_masked.mask_overlap(treat_cls_article, treat_cls_summary)
    if masked_result is None: return None, None
    input_ids = 'English Summary : ' + batch['EnglishSummary'] + ' %s Article : ' % treat_lingual + \
                masked_result['inputs']
    input_ids = tokenizer.batch_encode_plus([input_ids], max_length=2048, return_tensors='pt')['input_ids']
    labels = tokenizer.batch_encode_plus([masked_result['labels']], return_tensors='pt')['input_ids']
    if cuda_flag:
        input_ids, labels = input_ids.cuda(), labels.cuda()
    return input_ids, labels


if __name__ == '__main__':
    tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained('D:/PythonProject/mt5-small')
    if cuda_flag: model = model.cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1E-4)

    save_path = 'mT5_MKM_WikiLingual_Spanish&Portuguese/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    #############################################
    train_portuguese_data, test_portuguese_data = SeparateTrainTest(Loader_WikiLingual('portuguese'), 'portuguese')
    train_spanish_data, test_spanish_data = SeparateTrainTest(Loader_WikiLingual('spanish'), 'spanish')
    train_data, test_data = [], []
    train_data.extend(train_portuguese_data)
    train_data.extend(train_spanish_data)
    numpy.random.shuffle(train_data)
    test_data.extend(test_portuguese_data)
    test_data.extend(test_spanish_data)
    print('\nFinal Train Data has %d samples, Test Data has %d samples' % (len(train_data), len(test_data)))

    portuguese_masked, spanish_masked = MaskedKeywordsModule('portuguese'), MaskedKeywordsModule('spanish')
    ##############################################

    total_loss = 0.0
    step_counter = 0
    model.zero_grad()
    pbar = ProgressBar(n_total=20 * len(train_data))
    for epoch in range(20):
        for batch in train_data:
            if batch is None: continue
            step_counter += 1
            input_ids, labels = treatment(batch)
            if input_ids is None: continue

            loss = model.forward(input_ids=input_ids, labels=labels).loss
            loss.backward()
            total_loss += loss.data

            optimizer.step()
            model.zero_grad()
            pbar(step_counter, {'loss': loss.data})
            if step_counter % 1 == 0:
                print("\nstep: %7d\t loss: %7f\n" % (step_counter, total_loss))
                total_loss = 0.0

                with torch.set_grad_enabled(False):
                    val_pbar = ProgressBar(n_total=len(test_data))
                    for i, batch in enumerate(test_data):
                        input_ids, labels = treatment(batch)
                        if input_ids is None: continue

                        loss = model.forward(input_ids=input_ids, labels=labels).loss
                        val_pbar(i, {'loss': loss.data})
                        total_loss += loss.item()
                    print('\nVal Part Loss = ', total_loss)
                    with open(os.path.join(save_path, "log"), "a", encoding="UTF-8") as log:
                        log.write(
                            "%s\t step: %6d\t loss: %.2f\t \n" % (datetime.datetime.now(), step_counter, total_loss))

                    filename = "checkpoint-step-%06d" % step_counter
                    full_filename = os.path.join(save_path, filename)
                    model.save_pretrained(save_path + filename)

                total_loss = 0.0
