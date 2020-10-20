import argparse
import os
import sklearn
from train_and_test import train_and_test
from config_from_file import config_from_file
from dataset import TextDataset
from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config with parameters")
    args = parser.parse_args()
    cfg = config_from_file(args.config)
    dataset = TextDataset(cfg)

    predictions = []
    labels = []
    for i in tqdm(range(cfg.nfolds),desc='train_and_test'):
        preds, labs = train_and_test(cfg, dataset, fold=i)
        predictions.extend(preds)
        labels.extend(labs)

    cm = sklearn.metrics.confusion_matrix(labels, predictions, normalize='all')
    cm_msg=f' confusion matrix \n {cm} \n'
    print(cm_msg)
    acc = sklearn.metrics.accuracy_score(labels, predictions)
    acc_msg=f' accuracy \n {acc}\n'
    print(acc_msg)

    with open(os.path.join(cfg.work_dir,'metric.txt'), 'wt') as file:
        file.write(cm_msg)
        file.write(acc_msg)
