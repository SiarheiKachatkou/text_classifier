import argparse
import matplotlib.pyplot as plt
from collections import Counter
from config_from_file import config_from_file
from data import TextDataset

def text_corpus_to_words(text_corpus):
    words=[]
    for t in text_corpus:
        splits=t.split(' ')
        words.extend([s for s in splits if len(s)>2 ])
    return words

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config with parameters")
    args = parser.parse_args()
    cfg = config_from_file(args.config)

    dataset = TextDataset(cfg)

    word_sets = dict()
    pos_words=text_corpus_to_words(dataset.pos_text_corpus)
    neg_words=text_corpus_to_words(dataset.neg_text_corpus)
    for name, words in zip(['pos', 'neg'], [pos_words, neg_words]):

        plt.hist(words)
        plt.title(f'hist of {name} words')
        plt.show()

        counter = Counter(words)
        word_sets[name] = (set(words), counter)
        print(f'{name} descriptive stat:')
        most_common = counter.most_common()
        print(f'        most frequent words in {name}: ')
        for word, freq in most_common[:10]:
            print(f'            {word} {freq / len(words)}')

        print(f'        least frequent words in {name} ')
        for word, freq in most_common[-10:]:
            print(f'          {word} {freq / len(words)}')


    name1 = 'pos'
    name2 = 'neg'
    for _ in range(2):
        print(f'{name1} words absend in {name2}')
        diff_words = word_sets[name1][0] - word_sets[name2][0]
        list_diff_words = [(t, word_sets[name1][1].get(t)) for t in diff_words]
        list_diff_words.sort(key=lambda x: x[1], reverse=True)
        print([t[0] for t in list_diff_words])
        name1, name2 = name2, name1
