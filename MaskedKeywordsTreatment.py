import os


class MaskedKeywordsModule:
    def __init__(self, ignore_words_name, ignore_words_number=50):
        self.load_path = 'D:/PythonProject/Study202205/Pretreatment/'
        self.ignore_words = set()
        with open(os.path.join(self.load_path, ignore_words_name + 'IgnoreWords.txt'), 'r', encoding='UTF-8') as file:
            raw_data = file.readlines()
        for sample in raw_data[0:ignore_words_number]:
            self.ignore_words.add(sample.split(',')[0])

    def mask_overlap(self, article, summary, keywords_number=10):
        keywords_dictionary = {}
        for word in article.strip().split():
            if word in summary.strip().split():
                if word in self.ignore_words: continue
                if word in keywords_dictionary.keys():
                    keywords_dictionary[word] += 1
                else:
                    keywords_dictionary[word] = 1

        if len(keywords_dictionary) == 0: return None
        keywords_tuple = [[_, keywords_dictionary[_]] for _ in keywords_dictionary]

        keywords_tuple = sorted(keywords_tuple, key=lambda x: x[-1], reverse=True)[0:keywords_number]
        treat_keywords = set([_[0] for _ in keywords_tuple])
        treat_keywords = [_ for _ in treat_keywords]

        current_lm_sentence = ''
        for index in range(len(treat_keywords)):
            current_lm_sentence += '<extra_id_%d> ' % (index + 1) + treat_keywords[index] + ' '
        current_lm_sentence += '<extra_id_%d>' % (len(treat_keywords) + 1)

        current_article = article
        for index in range(len(treat_keywords)):
            current_article = current_article.replace(treat_keywords[index], '<extra_id_%d>' % (index + 1))
        return {'inputs': current_article, 'labels': current_lm_sentence}


if __name__ == '__main__':
    MaskedKeywordsModule(ignore_words_name='portuguese')
