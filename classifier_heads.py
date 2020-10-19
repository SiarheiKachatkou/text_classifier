from sklearn.ensemble import RandomForestClassifier
import abc
from abc import abstractmethod


class ClassifierHead(abc.ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, features, labels):
        raise NotImplementedError

    @abstractmethod
    def predict(self, sample):
        raise NotImplementedError


class RandomForestClassifierHead(ClassifierHead):

    def __init__(self, **kwargs):
        super().__init__()

        self._cl = RandomForestClassifier(**kwargs)

    def fit(self, features, labels):
        self._cl.fit(features, labels)

    def predict(self, features):
        return self._cl.predict(features)