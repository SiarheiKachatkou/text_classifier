import abc
from abc import abstractmethod
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
        return self._vectorizer.transform(text_tokenized)


class BERTFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def extract_feature(self, input):
        pass
