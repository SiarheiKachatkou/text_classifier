from feature_extractors import *
from classifier_heads import *


class Model(object):
    def __init__(self, feature_extractor: FeatureExtractor, classifier_head:ClassifierHead):
        self._feature_extractor=feature_extractor
        self._classifier_head=classifier_head

    def fit(self,dataset):
        features = self._feature_extractor.fit_transform(dataset.get_X())
        self._classifier_head.fit(features,dataset.get_labels())

    def predict(self,dataset):
        features = self._feature_extractor.fit_transform(dataset.get_X())
        predictions=self._classifier_head.predict(features)
        return predictions


def build_model(cfg):
    feature_extractor=eval(cfg.feature_extractor_str)
    classifier_head=eval(cfg.classifier_head_str)
    model=Model(feature_extractor,classifier_head)
    return model