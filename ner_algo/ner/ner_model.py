from .document import Document
from kashgari.utils import load_model
from collections import defaultdict


class KashgariModel(object):
    def __init__(self):
        pass

    def load_model(self, model_path):
        return load_model(model_path)


class Bert_BiLSTM_CRF_Model(KashgariModel):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model(model_path)

    def extract_company(self, title=None, content=None):
        entity = self.predict(title, content)
        return list(set([e[0] for e in entity['ORG'] if len(e[0]) > 1]))

    def predict(self, title=None, content=None):
        document = Document(title, content)
        data = [list(s.text.lower()) for s in document.sentences]
        preds = self.model.predict_entities(data)
        entity = self._format_result(document.sentences, preds)
        return entity
        
    def _format_result(self, sentences, preds):
        entity = defaultdict(list)
        for sentence, pred in zip(sentences, preds):
            for label in pred['labels']:
                value = sentence.text[label['start']: label['end'] + 1]
                entity_type = label['entity']
                entity[entity_type].append((value, sentence.idx, 
                                            label['start'], label['end'] + 1))
        return entity
                
