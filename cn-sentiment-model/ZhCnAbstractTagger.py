import DBSCAN
import bert_serving
import numpy as np
class ZhCNAbstractTagger(object):

    def __init__(self, master_ip):
        from bert_serving.client import BertClient
        self.bc = BertClient(ip=master_ip)

    @staticmethod
    def cleanse_text(text):
        t = text.replace(u'\u3000', '').replace('\r', '').replace('\n', '').replace('\\r', '').replace('\\n', '').lstrip("。")
        return re.sub('^[\u4e00-\u9fff]{2,4}，', '', t)

    @staticmethod
    def sent_tokenizer(texts):
        import re
        return [k for k in re.split('!|。|！|~|～', texts) if len(k) > 0]

    @staticmethod
    def extract_abstract_DBSCAN(embeddings):
        # DBSCAN
        clustering = DBSCAN(eps=8, min_samples=2).fit(embeddings)
        label_dict = defaultdict(list)
        idx = -1
        for label in clustering.labels_:
            idx = idx + 1
            if label != -1:
                label_dict[label].append(idx)
        selected_idx = []
        for label in label_dict:
            idxs = label_dict[label]
            c = np.mean(embeddings[idxs], axis=0)
            dist_vec = [np.linalg.norm(e - c) for e in embeddings]
            min_idx = np.argmin(dist_vec)
            selected_idx.append(min_idx)
        return selected_idx

    def extract_abstract(self, text, title):
        if text and len(text) > 0:# and len(title) > 0:
            sents = list(filter(lambda k: len(k) > 15, map(self.cleanse_text, self.sent_tokenizer(text))))[:100]
            if len(sents) == 0:
                return ""
            if title and len(title) > 0:
                try:
                    embeddings = self.bc.encode([title]+sents)
                    title_embed = embeddings[0]
                    embeddings = embeddings[1:]
                    dist_vec = [np.linalg.norm(e - title_embed) for e in embeddings]
                    top_k = np.array(dist_vec).argsort()[:3]
                except:
                    return ""
            else:
                try:
                    embeddings = self.bc.encode(sents)
                    top_k = [""]
                except:
                    return ""
            all_idxs = list(set([sorted(top_k)[0]] + self.extract_abstract_DBSCAN(embeddings)))
            return "".join(sents[k] for k in all_idxs)
        else:
            return ""