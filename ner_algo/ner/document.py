import re

class Document(object): 
    def __init__(self, title=None, content=None, max_len=4):
        self.title = title
        self.content = content
        self.max_len = max_len
        self.sentence_list = self.generate_sentence_list()
        
    def generate_sentence_list(self):
        sentence_list = []
        if self.title:
            s = Sentence(0, self.title)
            sentence_list.append(s)
        if self.content:
            s_list = cut_sentence(self.content)
            sentence_list += [Sentence(i+1, s_list[i]) for i in range(len(s_list))]
        return sentence_list

    @property
    def sentences(self):
        return self.sentence_list[:self.max_len]

    def __len__(self):
        return len(self.sentences)


class Sentence(object):
    def __init__(self, idx, text):
        self.idx = idx
        self.text = text

    def __repr__(self):
        return "{}".format(self.text)


def cut_sentence(content):
    content = content.replace("\n", '')
    # 单字符断句符
    content = re.sub('([。！？\?])([^”’])', r"\1\n\2", content)
    # 英文省略号
    content = re.sub('(\.{6})([^”’])', r"\1\n\2", content)
    # 中文省略号
    content = re.sub('(\…{2})([^”’])', r"\1\n\2", content)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    content = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', content)
    sentences = content.split("\n")
    # 只取有标点符号结尾的句子，过滤掉与正文不相关的文本
    valid_sentences = [s.strip() for s in sentences if len(s) > 1 and s[-1] in '。！？?”’']
    return valid_sentences    