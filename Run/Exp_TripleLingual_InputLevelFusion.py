import os
import torch
import numpy
import datetime
from Tools import ProgressBar
from Loader_WikiLingual import Loader_WikiLingual_Both, SeparateTrainTest
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from ModelStructure.mT5_Based_TripleLingual import MT5_TripleLingual_LateFusion_FinalEmbedding, MT5_TripleLingual_LateFusion_Project


def Treatment(batch):
    portuguese_article, spanish_article = batch['portugueseDocument'], batch['spanishDocument']
    english_summary = batch['EnglishSummary']
    if numpy.random.randint(2) == 0:
        input_ids = tokenizer.batch_encode_plus([
            'Summarize to English : Portuguese Document : ' + portuguese_article + ' Spanish Document : ' + spanish_article],
            return_tensors='pt')['input_ids']
    else:
        input_ids = tokenizer.batch_encode_plus([
            'Summarize to English : Spanish Document : ' + spanish_article + ' Portuguese Document : ' + portuguese_article],
            return_tensors='pt')['input_ids']
    summary_ids = tokenizer.batch_encode_plus([english_summary], return_tensors='pt')['input_ids']

    if cuda_flag:
        input_ids, summary_ids = input_ids.cuda(), summary_ids.cuda()
    return input_ids, summary_ids


cuda_flag = True
tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
if __name__ == '__main__':
    model = MT5ForConditionalGeneration.from_pretrained('D:/PythonProject/mt5-small')
    # model = MT5_TripleLingual_LateFusion.from_pretrained('D:/PythonProject/mt5-small')
    # model = MT5_TripleLingual_LateFusion_Project.from_pretrained('D:/PythonProject/mt5-small')
    if cuda_flag: model = model.cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1E-4)

    save_path = 'mT5_TripleLingual_InputLevelFusion/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    train_data, test_data = SeparateTrainTest(Loader_WikiLingual_Both('portuguese', 'spanish'), 'portuguese+spanish')
    total_loss = 0.0
    step_counter = 0
    model.zero_grad()
    pbar = ProgressBar(n_total=20 * len(train_data))
    for epoch in range(20):
        for batch in train_data:
            if batch is None: continue
            step_counter += 1
            input_ids, summary_ids = Treatment(batch)
            loss = model.forward(input_ids=input_ids, labels=summary_ids).loss
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
                        input_ids, summary_ids = Treatment(batch)
                        loss = model.forward(input_ids=input_ids, labels=summary_ids).loss

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
