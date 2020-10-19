import abc
from abc import abstractmethod
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor(abc.ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit_transform(self, text_corpus_tokenized):
        raise NotImplementedError

    @abstractmethod
    def transform(self, text_tokenized):
        raise NotImplementedError


class TFIDFFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()


    def fit_transform(self, text_corpus):

        self._vectorizer = TfidfVectorizer(lowercase=False)

        vectors = self._vectorizer.fit_transform(text_corpus)
        dense = vectors.todense()
        denselist = dense.tolist()
        return denselist

    def transform(self, text_tokenized):
        vectors=self._vectorizer.transform(text_tokenized)
        denselist=vectors.todense()
        return denselist


class BERTFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self._tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

    def extract_feature(self, input):
        pass


    def _tokenize(self, sent_list):

        sent_list = [self._tokenizer.encode(sent,add_special_tokens=False) for sent in sent_list]
        return sent_list

