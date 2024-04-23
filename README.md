---
tags:
- keyword-extraction
language:
- vi
---


# <a name="introduction"></a>  KeyBERTVi - Keyword Extraction for Vietnamese language

Inspired by [KeyBERT](https://github.com/MaartenGr/KeyBERT), KeyBERTVi implements a similar keyword extraction technique that leverages the embeddings of [PhoBERT](https://huggingface.co/vinai/phobert-base) and minimal linguistics properties to extract keywords and keyphrases that are most similar to the document.

<a name="toc"/></a>
## Table of Contents  
<!--ts-->  
   1. [About the Project](#about)  
   2. [Getting Started](#gettingstarted)  
        2.1. [Installation](#installation)  
        2.2. [Basic Usage](#usage)  
        2.3. [Diversify Results](#diversify)  
   3. [Limitations](#limitations)  
<!--te-->  

<a name="about"/></a>
## 1. About the Project

This implementation took inspiration from the simple yet intuitive and powerful method of [KeyBERT](https://github.com/MaartenGr/KeyBERT/), applied for the Vietnamese language. PhoBERT are used to generate both document-level embeddings and word-level embeddings for extracted N-grams. Cosine similarity is then used to compute which N-grams are most similar to the document-level embedding, thus can be perceived as most representative of the document. 
Preprocessing catered to the Vietnamese language was applied. 

Test with your own documents at [KeyBERTVi Space](https://huggingface.co/spaces/tpha4308/keybertvi-app). 

<a name="gettingstarted"/></a>
## 2. Getting Started
<a name="installation"/></a>
###  2.1. Setting up

```bash
  git clone https://huggingface.co/tpha4308/keyword-extraction-viet
```

You can use existing pre-trained models in the repo or download your own and put them in `pretrained-models` folder. 

```python
  phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
  phobert.eval()
  torch.save(phobert, f'{dir_path}/pretrained-models/phobert.pt')
  
  ner_model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
  ner_model.eval()
  torch.save(ner_model, f'{dir_path}/pretrained-models/ner-vietnamese-electra-base.pt')
```

**Note:** `dir_path` is the absolute path to the repo. 

As [PhoBERT](https://huggingface.co/vinai/phobert-base) requires [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) as part of pre-processing, the folder `pretrained-models/vncorenlp` is required. To download your own: 
```bash
  pip install py_vncorenlp
```

```python
  import py_vncorenlp
  
  py_vncorenlp.download_model(save_dir=f'{dir_path}/pretrained-models/vncorenlp')
```

<a name="usage"/></a>
###  2.2. Basic Usage

```python
  phobert = torch.load(f'{dir_path}/pretrained-models/phobert.pt')
  phobert.eval()
  ner_model = torch.load(f'{dir_path}/pretrained-models/ner-vietnamese-electra-base.pt')
  ner_model.eval()
  kw_pipeline = KeywordExtractorPipeline(phobert, ner_model)
```

```python
  title = "Truyền thuyết và hiện tại Thành Cổ Loa"
  text = """
            Nhắc đến Cổ Loa, người ta nghĩ ngay đến truyền thuyết về An Dương Vương được thần Kim Quy bày cho cách xây thành, về chiếc lẫy nỏ thần làm từ móng chân rùa thần và mối tình bi thương Mỵ Châu – Trọng Thủy. Đằng sau những câu chuyện thiên về tâm linh ấy, thế hệ con cháu còn khám phá được những giá trị khảo cổ to lớn của Cổ Loa.
            Khu di tích Cổ Loa cách trung – tâm Hà Nội 17km thuộc huyện Đông Anh, Hà Nội, có diện tích bảo tồn gần 500ha được coi là địa chỉ văn hóa đặc biệt của thủ đô và cả nước. Cổ Loa có hàng loạt di chỉ khảo cổ học đã được phát hiện, phản ánh quá trình phát triển liên tục của dân tộc ta từ sơ khai qua các thời kỳ đồ đồng, đồ đá và đồ sắt mà đỉnh cao là văn hóa Đông Sơn, vẫn được coi là nền văn minh sông Hồng thời kỳ tiền sử của dân tộc Việt Nam.
            Cổ Loa từng là kinh đô của nhà nước Âu Lạc thời kỳ An Dương Vương (thế kỷ III TCN) và của nước Đại Việt thời Ngô Quyền (thế kỷ X) mà thành Cổ Loa là một di tích minh chứng còn lại cho đến ngày nay. Thành Cổ Loa được các nhà khảo cổ học đánh giá là “tòa thành cổ nhất, quy mô lớn vào bậc nhất, cấu trúc cũng thuộc loại độc đáo nhất trong lịch sử xây dựng thành lũy của người Việt cổ”.
          """
  inp = {"title": title, "text": text}
  kws = kw_pipeline(inputs=inp, min_freq=1, ngram_n=(1, 3), top_n=5, diversify_result=False)

  [('Khu di_tích Cổ_Loa', 0.88987315),
  ('Âu_Lạc thời_kỳ An_Dương_Vương', 0.8680505),
  ('thành Cổ_Loa', 0.8661723),
  ('hàng_loạt di_chỉ khảo_cổ_học', 0.8644231),
  ('lịch_sử xây_dựng thành_luỹ', 0.8375939)]
```

<a name="diversify"/></a>
###  2.3. Diversify Results

More information needed

<a name="limitations"/></a>
## 3. Limitations

More information needed

## References
1. https://github.com/MaartenGr/KeyBERT
2. https://github.com/VinAIResearch/PhoBERT
3. https://huggingface.co/NlpHUST/ner-vietnamese-electra-base
4. https://github.com/undertheseanlp/underthesea
5. https://github.com/vncorenlp/VnCoreNLP
