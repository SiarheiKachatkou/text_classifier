import abc
from abc import abstractmethod
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

class FeatureExtractor(abc.ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit_transform(self, text_corpus_tokenized):
        raise NotImplementedError

    @abstractmethod
    def transform(self, text_tokenized):
        raise NotImplementedError


class TFIDFFeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__()
        self._vectorizer = TfidfVectorizer(**kwargs)

    def fit_transform(self, text_corpus):
        vectors = self._vectorizer.fit_transform(text_corpus)
        dense = vectors.todense()
        denselist = dense.tolist()
        return denselist

    def transform(self, text_list):
        vectors=self._vectorizer.transform(text_list)
        denselist=vectors.todense()
        return denselist


class BERTFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self._model.eval()


    def _text_to_tokens_ids(self, text_list):
        ids=[]
        for text in text_list:
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = self._tokenizer.tokenize(marked_text)
            indexed_tokens = self._tokenizer.convert_tokens_to_ids(tokenized_text)
            ids.append(indexed_tokens)
        return ids

    def fit_transform(self, text_corpus):

        return self.transform(text_corpus)

    def transform(self, text_list):
        tokenized_texts = self._text_to_tokens_ids(text_list)
        segments_ids = [[1] * len(tokenized_text) for tokenized_text in tokenized_texts]

        sentence_embeddings = []
        with torch.no_grad():
            for tokenized_text, segments_id in zip(tokenized_texts, segments_ids):
                tokens_tensor = torch.tensor([tokenized_text])
                segments_tensors = torch.tensor([segments_id])
                outputs = self._model(tokens_tensor, segments_tensors)

                # Evaluating the model will return a different number of objects based on
                # how it's  configured in the `from_pretrained` call earlier. In this case,
                # becase we set `output_hidden_states = True`, the third item will be the
                # hidden states from all layers. See the documentation for more details:
                # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
                hidden_states = outputs[2]
                hidden_states = torch.cat(hidden_states[1:], dim=1)

                # Calculate the average of all token vectors.
                sentence_embedding = torch.mean(hidden_states, dim=1)[0].detach().cpu().numpy()
                sentence_embeddings.append(sentence_embedding)

        return sentence_embeddings


