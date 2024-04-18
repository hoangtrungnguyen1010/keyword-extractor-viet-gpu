import py_vncorenlp
from transformers import AutoTokenizer, pipeline
import torch
import os
from keyword_extraction import extract_keywords
import sys


class KeyBERTVi:

    def __init__(self, stopwords_file_path):
        self.annotator = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos"],
                                                save_dir=f'{dir_path}/pretrained-models/vncorenlp')
        # model = py_vncorenlp.VnCoreNLP(save_dir='/absolute/path/to/vncorenlp')
        print("Loading PhoBERT model")
        self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

        # use absolute path because torch is cached
        self.phobert = torch.load(f'{dir_path}/pretrained-models/phobert.pt')
        self.phobert.eval()

        print("Loading NER model")
        ner_tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
        ner_model = torch.load(f'{dir_path}/pretrained-models/ner-vietnamese-electra-base.pt')
        ner_model.eval()
        self.ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)

        with open(stopwords_file_path) as f:
            self.stopwords = [w.strip() for w in f.readlines()]

    def extract_keywords(self, title, text, ngram_range=(1, 3), top_n=5):
        keyword_ls = extract_keywords(text, title,
                                      self.ner_pipeline,
                                      self.annotator,
                                      self.phobert_tokenizer,
                                      self.phobert,
                                      self.stopwords,
                                      ngram_n=ngram_range,
                                      top_n=top_n)
        return keyword_ls


dir_path = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    # args
    print(dir_path)

    stopwords_file_path = f'{dir_path}/vietnamese-stopwords-dash.txt'

    text_file_path = sys.argv[1]
    with open(f'{dir_path}/{text_file_path}', 'r') as f:
        text = ' '.join([ln.strip() for ln in f.readlines()])
        print(text)

    kw_model = KeyBERTVi(stopwords_file_path)
    title = None
    keyword_ls = kw_model.extract_keywords(title, text, ngram_range=(1, 3), top_n=5)
    print(keyword_ls)
