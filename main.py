import argparse
import torch.multiprocessing as mp
import numpy as np
import sklearn
from train_and_test import train_and_test
from config_from_file import config_from_file
from dataset import TextDataset


def run(i,q, model_fn, dataset):
    q.put(train_and_test(model_fn, dataset, fold=i))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config with parameters")
    args = parser.parse_args()
    cfg = config_from_file(args.config)
    dataset = TextDataset(cfg)

    if cfg.is_debug: #debug multiprocess code is pain
        for i in range(cfg.nfolds):
            predictions, labels = train_and_test(cfg, dataset, fold=i)
    else:
        n_procs = mp.num_cpus()
        q=mp.Queue(n_procs)

        processes = []
        for rank in range(n_procs):
            p = mp.Process(target=run, args=(rank,q, cfg, dataset))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        predictions = []
        labels = []
        for _ in range(n_procs):
            preds, labs = q.get()
            predictions.extend(preds)
            labels.extend(labs)

    cm=sklearn.metrics.confusion_matrix(labels, predictions)
    print(f' confusion matrix {cm}')
    acc=sklearn.metrics.accuracy_score(labels, predictions)
    print(f' accuracy {acc}')