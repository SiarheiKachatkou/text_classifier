

class ClassifierHead(object):
    def __init__(self):
        pass

    def fit(self,dataset):
        raise NotImplementedError

    def predict(self, sample):
        raise NotImplementedError