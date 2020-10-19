import torch
import numpy as np

class TextDataset(object):
    def __init__(self, pos_text_corpus_path, neg_text_corpus_path ):
        self._tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        self._load_corpus('_pos_text_corpus',pos_text_corpus_path)
        self._load_corpus('_neg_text_corpus', neg_text_corpus_path)

    def _tokenize(self, sent_list):

        sent_list = [self._tokenizer.encode(sent,add_special_tokens=False) for sent in sent_list]
        return sent_list

    def _load_corpus(self, attr_name, path_to_corpus):
        with open(path_to_corpus, 'rt') as file:
            sent_list = file.readlines()
            sent_list = self._tokenize(sent_list)
            setattr(self, attr_name, sent_list)

    @property
    def pos_text_corpus(self):
        return self._pos_text_corpus

    @property
    def neg_text_corpus(self):
        return self._neg_text_corpus

    def untokenize(self, token_or_list):
        if not isinstance(token_or_list, (list,np.ndarray)):
            tokens_list=[token_or_list]
        else:
            tokens_list=token_or_list

        return  self._tokenizer.decode(tokens_list)









