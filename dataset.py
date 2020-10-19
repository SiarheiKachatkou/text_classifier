import numpy as np
import copy
from sklearn.model_selection import KFold


class TextDataset(object):
    def __init__(self, config ):

        self._cfg = config
        self.pos_text_corpus=self._load_corpus(config.pos_corpus_path)
        self.neg_text_corpus=self._load_corpus(config.neg_corpus_path)


        self._build_fold_splits()

    def _build_fold_splits(self):
        kf = KFold(n_splits=self._cfg.nfolds, random_state=self._cfg.random_seed, shuffle=True)
        labels = self.get_labels()
        self._sp = list(kf.split(labels))

    def _load_corpus(self, path_to_corpus):
        with open(path_to_corpus, 'rt') as file:
            sent_list = file.readlines()
            if self._cfg.is_debug:
                sent_list=sent_list[:100]
            return sent_list

    def get_X(self):
        return np.array(self.pos_text_corpus+self.neg_text_corpus)

    def get_labels(self):
        return np.array([1]*len(self.pos_text_corpus)+[0]*len(self.neg_text_corpus))

    def get_fold(self, fold):

        train_idxs, val_idxs = self._sp[fold]
        train_dataset = self.get_subset(train_idxs)
        val_dataset = self.get_subset(val_idxs)

        return train_dataset, val_dataset

    def get_subset(self, idxs):
        x=self.get_X()
        labels=self.get_labels()
        x=x[idxs]
        labels=labels[idxs]

        subset = copy.copy(self)
        subset.pos_text_corpus=list(x[labels==1])
        subset.neg_text_corpus=list(x[labels == 0])
        return subset











