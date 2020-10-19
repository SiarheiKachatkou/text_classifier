import argparse
import torch.multiprocessing as mp
import numpy as np
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
            train_and_test(cfg, dataset, fold=i)
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
        results=[]
        for _ in range(n_procs):
            results.append(np.array(q.get()))
        results=np.stack(results,axis=0)
        results=np.mean(results,axis=0)
        print(results)