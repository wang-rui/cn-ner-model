import numpy as np
class SentimentClf:
    def __init__(self, clf_model, key_dict, label_list):
        self.model = clf_model
        self.dict = key_dict
        self.label_cls = label_list

    def get_bow_encoding(self, bow_corpus):

        bow_encoding = np.zeros([max(1, len(bow_corpus)), len(self.dict)])
        for idx, line in enumerate(bow_corpus):
            for item in line:
                bow_encoding[idx][item[0]] = 1
        return bow_encoding

    def predict_sentiment(self, txts):
        #         txt = '刘士余, 接替, 任命, 董事长, 担任, 主席, 中国'
        if isinstance(txts, list):
            bow_corpus = [self.dict.doc2bow(doc.split(', ')) for doc in txts]
        else:
            bow_corpus = [self.dict.doc2bow(txts.split(','))]
        bow_encoding = self.get_bow_encoding(bow_corpus)
        clf_data = self.model.predict(np.mat(bow_encoding))
        if len(clf_data):
            return [self.get_proba(int(x)) for x in clf_data]
        else:
            return [0]

    def get_proba(self, l_cls):
        return np.random.uniform(self.label_cls[l_cls].left, self.label_cls[l_cls].right)
