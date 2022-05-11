import json
import tqdm

if __name__ == '__main__':
    appoint_lingual = 'portuguese'
    keywords = json.load(open(appoint_lingual + 'Keywords.json', 'r'))
    total_keywords = {}
    for sample in tqdm.tqdm(keywords):
        for word in sample.keys():
            if word in total_keywords.keys():
                total_keywords[word] += sample[word]
            else:
                total_keywords[word] = sample[word]

    keywords_result = [[_, total_keywords[_]] for _ in total_keywords.keys()]
    keywords_result = sorted(keywords_result, key=lambda x: x[-1], reverse=True)
    with open(appoint_lingual + 'IgnoreWords.txt', 'w', encoding='UTF-8') as file:
        for index in range(len(keywords_result)):
            file.write(keywords_result[index][0] + ',' + str(keywords_result[index][1]) + '\n')
