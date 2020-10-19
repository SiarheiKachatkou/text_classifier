import argparse
import matplotlib.pyplot as plt
from collections import Counter
from config_from_file import config_from_file
from data import TextDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config with parameters")
    args = parser.parse_args()
    cfg = config_from_file(args.config)

    dataset = TextDataset(cfg)

    token_sets = dict()

    for name, tokenized_corpus in zip(['pos', 'neg'], [dataset.pos_text_corpus, dataset.neg_text_corpus]):

        tokens = []
        avg_length = 0
        for t in tokenized_corpus:
            avg_length += len(t)
            tokens.extend(t)
        avg_length /= len(tokenized_corpus)
        plt.hist(tokens)
        plt.title(f'hist of {name} tokens')
        plt.show()

        counter = Counter(tokens)
        token_sets[name] = (set(tokens), counter)
        print(f'{name} descriptive stat:')
        most_common = counter.most_common()
        print(f'        most frequent words in {name}: ')
        for token, freq in most_common[:10]:
            print(f'            {dataset.untokenize(token)} {freq / len(tokens)}')

        print(f'        least frequent words in {name} ')
        for token, freq in most_common[-10:]:
            print(f'          {dataset.untokenize(token)} {freq / len(tokens)}')

        print(f'        average sent length {avg_length}')

    name1 = 'pos'
    name2 = 'neg'
    for _ in range(2):
        print(f'{name1} words absend in {name2}')
        diff_tokens = token_sets[name1][0] - token_sets[name2][0]
        list_diff_tokens = [(t, token_sets[name1][1].get(t)) for t in diff_tokens]
        list_diff_tokens.sort(key=lambda x: x[1], reverse=True)
        print(dataset.untokenize([t[0] for t in list_diff_tokens]))
        name1, name2 = name2, name1
