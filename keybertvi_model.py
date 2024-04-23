# import py_vncorenlp
# from transformers import AutoTokenizer, pipeline
# import torch
# import os
# from model.keyword_extraction_utils import extract_keywords
#
#
# class KeyBERTVi:
#
#     def __init__(self, stopwords_file_path=None):
#         self.annotator = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos"],
#                                                 save_dir=f'{dir_path}/pretrained-models/vncorenlp')
#         # model = py_vncorenlp.VnCoreNLP(save_dir='/absolute/path/to/vncorenlp')
#         print("Loading PhoBERT model")
#         self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
#
#         # use absolute path because torch is cached
#         self.phobert = torch.load(f'{dir_path}/pretrained-models/phobert.pt')
#         self.phobert.eval()
#
#         print("Loading NER model")
#         ner_tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
#         ner_model = torch.load(f'{dir_path}/pretrained-models/ner-vietnamese-electra-base.pt')
#         ner_model.eval()
#         self.ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
#
#         if stopwords_file_path is None:
#             stopwords_file_path = f'{dir_path}/vietnamese-stopwords-dash.txt'
#         with open(stopwords_file_path) as f:
#             self.stopwords = [w.strip() for w in f.readlines()]
#
#     def extract_keywords(self, title, text, ngram_range=(1, 3), top_n=5, use_kmeans=False, use_mmr=False, min_freq=1):
#         keyword_ls = extract_keywords(text, title,
#                                       self.ner_pipeline,
#                                       self.annotator,
#                                       self.phobert_tokenizer,
#                                       self.phobert,
#                                       self.stopwords,
#                                       ngram_n=ngram_range,
#                                       top_n=top_n,
#                                       use_kmeans=use_kmeans,
#                                       use_mmr=use_mmr,
#                                       min_freq=min_freq)
#         return keyword_ls
#
#     def highlight(self, text, keywords):
#         kw_ls = [' '.join(kw.split('_')) for kw, score in keywords]
#         for key in kw_ls:
#             text = text.replace(f" {key}", f" <mark>{key}</mark>")
#         return text
#
#
# dir_path = os.path.dirname(os.path.realpath(__file__))
# if __name__ == "__main__":
#     # args
#     # print(dir_path)
#
#     stopwords_file_path = f'{dir_path}/vietnamese-stopwords-dash.txt'
#
#     # text_file_path = sys.argv[1]
#     # with open(f'{dir_path}/{text_file_path}', 'r') as f:
#     #     text = ' '.join([ln.strip() for ln in f.readlines()])
#         # print(text)
#
#     # kw_model = KeyBERTVi()
#     # model_name_on_hub = "KeyBERTVi"
#     # kw_model.save_pretrained(model_name_on_hub)
#     # kw_model.phobert_tokenizer.save_pretrained(model_name_on_hub)
#
#     # title = None
#     # keyword_ls = kw_model.extract_keywords(title, text, ngram_range=(1, 3), top_n=5)
#     # print(keyword_ls)
