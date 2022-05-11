import json
import tqdm
from Loader_WikiLingual import Loader_WikiLingual

if __name__ == '__main__':
    appoint_lingual = 'portuguese'

    total_keywords = []
    treat_data = Loader_WikiLingual(target_lingual=appoint_lingual)
    for treat_sample in tqdm.tqdm(treat_data):
        summary = treat_sample['CrossLingualSummary'].strip().split()
        article = treat_sample['CrossLingualDocument'].strip().split()

        dictionary = {}
        for word in article:
            if word not in summary: continue
            if word in dictionary.keys():
                dictionary[word] += 1
            else:
                dictionary[word] = 1

        total_keywords.append(dictionary)
    json.dump(total_keywords, open(appoint_lingual + 'Keywords.json', 'w'))
