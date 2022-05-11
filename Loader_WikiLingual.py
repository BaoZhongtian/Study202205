import os
import pickle
import tqdm
import json


def Loader_WikiLingual(target_lingual):
    load_path = 'D:/ProjectData/pkl/pkl/'
    english_data = pickle.load(open(load_path + 'english.pkl', 'rb'))
    target_data = pickle.load(open(load_path + '%s.pkl' % target_lingual, 'rb'))

    total_sample = []
    for treat_sample in tqdm.tqdm(list(target_data.items())):
        cross_lingual_url = treat_sample[0]
        treat_sample = treat_sample[1]
        for section_name in treat_sample:
            english_correlated = english_data[treat_sample[section_name]['english_url']][
                treat_sample[section_name]['english_section_name']]

            current_sample = {'EnglishSummary': english_correlated['summary'],
                              'EnglishDocument': english_correlated['document'],
                              'EnglishSectionName': treat_sample[section_name]['english_section_name'],
                              'EnglishURL': treat_sample[section_name]['english_url'],
                              'CrossLingualSummary': treat_sample[section_name]['summary'],
                              'CrossLingualDocument': treat_sample[section_name]['document'],
                              'CrossLingualSectionName': section_name,
                              'CrossLingualURL': cross_lingual_url, 'CrossLingualName': target_lingual}
            total_sample.append(current_sample)
    print('\n')
    print('Appoint Lingual is', target_lingual)
    print('Total', len(total_sample), 'Samples')
    return total_sample


def Loader_WikiLingual_Both(lingual_x, lingual_y):
    result_x = Loader_WikiLingual(lingual_x)
    result_y = Loader_WikiLingual(lingual_y)

    english_url = {}
    for index, treat_sample in enumerate(result_x):
        english_url[treat_sample['EnglishURL']] = index

    total_sample = []
    for treat_sample in result_y:
        if treat_sample['EnglishURL'] not in english_url: continue
        related_index = english_url[treat_sample['EnglishURL']]

        total_sample.append(
            {'EnglishSummary': treat_sample['EnglishSummary'], 'EnglishDocument': treat_sample['EnglishDocument'],
             'EnglishSectionName': treat_sample['EnglishSectionName'], 'EnglishURL': treat_sample['EnglishURL'],
             '%sSummary' % lingual_x: result_x[related_index]['CrossLingualSummary'],
             '%sDocument' % lingual_x: result_x[related_index]['CrossLingualDocument'],
             '%sSectionName' % lingual_x: result_x[related_index]['CrossLingualSectionName'],
             '%sURL' % lingual_x: result_x[related_index]['CrossLingualURL'],

             '%sSummary' % lingual_y: treat_sample['CrossLingualSummary'],
             '%sDocument' % lingual_y: treat_sample['CrossLingualDocument'],
             '%sSectionName' % lingual_y: treat_sample['CrossLingualSectionName'],
             '%sURL' % lingual_y: treat_sample['CrossLingualURL']})

    print('\n')
    print('Appoint Lingual is', lingual_x, lingual_y)
    print('Total', len(total_sample), 'Samples')
    return total_sample


def SeparateTrainTest(treat_data, treat_ids_name):
    load_path = 'D:/PythonProject/Study202205/Pretreatment/'
    appoint_ids = json.load(open(load_path + treat_ids_name + 'RandomSeparateIds.json', 'r'))
    train_ids, test_ids = appoint_ids['train_ids'], appoint_ids['test_ids']

    train_part, test_part = [treat_data[_] for _ in train_ids], [treat_data[_] for _ in test_ids]
    print('Separate Train Test')
    print('Train Part %d Samples' % len(train_part))
    print('Test Part %d Samples' % len(test_part))
    return train_part, test_part


if __name__ == '__main__':
    # Loader_WikiLingual_MaskedKeywords_Overlap(target_lingual='portuguese')
    SeparateTrainTest(Loader_WikiLingual('portuguese'), 'portuguese')
