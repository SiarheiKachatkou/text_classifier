

class TextDataset(object):
    def __init__(self, pos_text_corpus_path, neg_text_corpus_path ):

        with open(pos_text_corpus_path,'rt') as file:
            self._pos_text_corpus = file.readlines()

        with open(neg_text_corpus_path, 'rt') as file:
            self._neg_text_corpus = file.readlines()




