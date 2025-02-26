import py_vncorenlp
from transformers import AutoTokenizer, Pipeline, pipeline
import os
import torch

from model.keyword_extraction_utils import *
from model.process_text import process_text_pipeline

dir_path = os.path.dirname(os.path.realpath(__file__))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class KeywordExtractorPipeline(Pipeline):
    def __init__(self, model, ner_model, verbose = False, **kwargs):
        super().__init__(model, **kwargs)
        self.annotator = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos"],
                                                save_dir=f'{dir_path}/pretrained-models/vncorenlp')
        self.verbose = verbose
        if self.verbose:
            print("Loading PhoBERT tokenizer")
        self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.phobert = model.to(device)

        if self.verbose:
            print("Loading NER tokenizer")
        ner_tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
        self.ner_pipeline = pipeline("ner", model=ner_model.to(device), tokenizer=ner_tokenizer, device=0 if torch.cuda.is_available() else -1)

        stopwords_file_path = f'{dir_path}/vietnamese-stopwords-dash.txt'
        with open(stopwords_file_path) as f:
            self.stopwords = [w.strip() for w in f.readlines()]

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}

        for possible_preprocess_kwarg in ["text", "title"]:
            if possible_preprocess_kwarg in kwargs:
                preprocess_kwargs[possible_preprocess_kwarg] = kwargs[possible_preprocess_kwarg]

        for possible_forward_kwarg in ["ngram_n", "min_freq"]:
            if possible_forward_kwarg in kwargs:
                forward_kwargs[possible_forward_kwarg] = kwargs[possible_forward_kwarg]

        for possible_postprocess_kwarg in ["top_n", "diversify_result"]:
            if possible_postprocess_kwarg in kwargs:
                postprocess_kwargs[possible_postprocess_kwarg] = kwargs[possible_postprocess_kwarg]

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs):
        title = process_text_pipeline(inputs['title']) if inputs['title'] else None
        text = process_text_pipeline(inputs['text'])
        return {"text": text, "title": title}

    def _forward(self, model_inputs, ngram_n, min_freq):
        text = model_inputs['text']
        title = model_inputs['title']

        # Getting segmentized document
        ne_ls, doc_segmentised = get_segmentised_doc(self.ner_pipeline, self.annotator, title, text)
        filtered_doc_segmentised = compute_filtered_text(self.annotator, title, text)

        doc_embedding = get_doc_embeddings(filtered_doc_segmentised, self.phobert_tokenizer, self.phobert, self.stopwords).to(device)

        ngram_list = self.generate_ngram_list(doc_segmentised, filtered_doc_segmentised, ne_ls, ngram_n, min_freq)
        
        if self.verbose:
            print("Final ngram list", sorted(ngram_list))

        ngram_embeddings = compute_ngram_embeddings(self.phobert_tokenizer, self.phobert, ngram_list)

        return {"ngram_list": ngram_list, "ngram_embeddings": ngram_embeddings, "doc_embedding": doc_embedding}

    def postprocess(self, model_outputs, top_n, diversify_result):
        ngram_list = model_outputs['ngram_list']
        ngram_embeddings = model_outputs['ngram_embeddings']
        doc_embedding = model_outputs['doc_embedding']

        ngram_result = self.extract_keywords(doc_embedding, ngram_list, ngram_embeddings)
        non_diversified = sorted([(ngram, ngram_result[ngram]) for ngram in ngram_result], key=lambda x: x[1], reverse=True)[:top_n]

        if diversify_result:
            return diversify_result_kmeans(ngram_result, ngram_embeddings, top_n=top_n)
        return non_diversified

    def generate_ngram_list(self, doc_segmentised, filtered_doc_segmentised, ne_ls, ngram_n, min_freq, verbose):
        ngram_low, ngram_high = ngram_n

        ngram_list = set()
        for n in range(ngram_low, ngram_high + 1):
            ngram_list.update(get_candidate_ngrams(doc_segmentised, filtered_doc_segmentised, n, self.stopwords))

        ne_ls_segmented = [self.annotator.word_segment(ne)[0] for ne in ne_ls]
        
        if self.verbose:
            print("Named Entities list", ne_ls_segmented)
            
        ngram_list.update(ne_ls_segmented)

        ngram_list = remove_overlapping_ngrams(ngram_list)

        if min_freq > 1:
            ngram_list = limit_minimum_frequency(doc_segmentised, ngram_list, min_freq=min_freq)
            return ngram_list.keys()

        return ngram_list

    def extract_keywords(self, doc_embedding, ngram_list, ngram_embeddings):
        ngram_result = compute_ngram_similarity(ngram_list, ngram_embeddings, doc_embedding)
        ngram_result = remove_duplicates(ngram_result)
        return ngram_result


if __name__ == "__main__":
    phobert = torch.load(f'{dir_path}/pretrained-models/phobert.pt', map_location=device)
    phobert.to(device).eval()
    ner_model = torch.load(f'{dir_path}/pretrained-models/ner-vietnamese-electra-base.pt', map_location=device)
    ner_model.to(device).eval()

    kw_pipeline = KeywordExtractorPipeline(phobert, ner_model)

    text_file_path = f'{dir_path}/test_file.txt'
    with open(text_file_path, 'r') as f:
        text = ' '.join([ln.strip() for ln in f.readlines()])

    inp = {"text": text, "title": None}
    kws = kw_pipeline(inputs=inp, min_freq=1, ngram_n=(1, 3), top_n=5, diversify_result=False)
    print(kws)

