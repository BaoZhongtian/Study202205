import os
import torch
import numpy
import datetime
from Tools import ProgressBar
from Run.Exp_TripleLingual import Treatment
from Loader_WikiLingual import Loader_WikiLingual_Both, SeparateTrainTest
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from ModelStructure.mT5_Based_TripleLingual import MT5_TripleLingual_LateFusion_FinalEmbedding, \
    MT5_TripleLingual_LateFusion_Project,MT5_TripleLingual_HiddenLayerProject

cuda_flag = True
tokenizer = MT5Tokenizer.from_pretrained('D:/PythonProject/mt5-small')
if __name__ == '__main__':
    # model = MT5_TripleLingual_LateFusion.from_pretrained('D:/PythonProject/mt5-small')
    model = MT5_TripleLingual_HiddenLayerProject.from_pretrained('D:/PythonProject/mt5-small')

    if cuda_flag: model = model.cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1E-4)

    save_path = 'mT5_TripleLingual_HiddenStateProject/'
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
            portuguese_input_ids, spanish_input_ids, summary_ids = Treatment(batch)
            loss = model.forward(input_ids_x=portuguese_input_ids, input_ids_y=spanish_input_ids,
                                 labels=summary_ids).loss
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
                        portuguese_input_ids, spanish_input_ids, summary_ids = Treatment(batch)
                        loss = model.forward(input_ids_x=portuguese_input_ids, input_ids_y=spanish_input_ids,
                                             labels=summary_ids).loss

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
