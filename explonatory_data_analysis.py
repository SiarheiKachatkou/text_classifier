import argparse
from config_from_file import config_from_file
from data import TextDataset

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config with parameters" )
    args=parser.parse_args()
    cfg=config_from_file(args.config)

    dataset=TextDataset(cfg.pos_corpus_path, cfg.neg_corpus_path)

    dbg=1