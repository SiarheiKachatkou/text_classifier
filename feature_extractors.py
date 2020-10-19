

class FeatureExtractor(object):
    def __init__(self):
        pass

    def extract_feature(self, input):
        raise NotImplementedError


class TFIDFFeatureExtractor(FeatureExtractor):
    def __init__(self, text_corpuse):
        super().__init__()

    def extract_feature(self, input):
        pass

class BERTFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def extract_feature(self, input):
        pass
