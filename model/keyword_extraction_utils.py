from string import punctuation
import numpy as np
import torch
from sklearn.cluster import KMeans
from model.named_entities import get_named_entities
import math

punctuation = [c for c in punctuation if c != "_"]
punctuation += ["“", "–", ",", "…", "”", "–"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


ethnicity_dict_map = {"H'Mông": "HMông",
                      "H'mông": "HMông",
                      "H’mông": "HMông",
                      "H’Mông": "HMông",
                      "H’MÔNG": "HMông",
                      "M'Nông": "MNông",
                      "M'nông": "MNông",
                      "M'NÔNG": "MNông",
                      "M’Nông": "MNông",
                      "M’NÔNG": "MNông",
                      "K’Ho": "KHo",
                      "K’Mẻo": "KMẻo"}


def sub_sentence(sentence):
    sent = []

    start_index = 0
    while start_index < len(sentence):
        idx_list = []
        for p in punctuation:
            idx = sentence.find(p, start_index)
            if idx != -1:
                idx_list.append(idx)

        if len(idx_list) == 0:
            sent.append(sentence[start_index:].strip())
            break

        end_index = min(idx_list)

        subsent = sentence[start_index:end_index].strip()
        if len(subsent) > 0:
            sent.append(subsent)

        start_index = end_index + 1

    return sent


def check_for_stopwords(ngram, stopwords_ls):
    for ngram_elem in ngram.split():
        for w in stopwords_ls:
            if ngram_elem == w:  # or ngram_elem.lower() == w:
                return True
    return False


def compute_ngram_list(segmentised_doc, ngram_n, stopwords_ls, subsentences=True):
    if subsentences:
        output_sub_sentences = []
        for sentence in segmentised_doc:
            output_sub_sentences += sub_sentence(sentence)
    else:
        output_sub_sentences = segmentised_doc

    ngram_list = []
    for sentence in output_sub_sentences:
        sent = sentence.split()
        for i in range(len(sent) - ngram_n + 1):
            ngram = ' '.join(sent[i:i + ngram_n])
            if ngram not in ngram_list and not check_for_stopwords(ngram, stopwords_ls):
                ngram_list.append(ngram)

    final_ngram_list = []
    for ngram in ngram_list:
        contains_number = False
        for char in ngram:
            if char.isnumeric():
                contains_number = True
                break
        if not contains_number:
            final_ngram_list.append(ngram)

    return final_ngram_list


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_doc_embeddings(segmentised_doc, tokenizer, phobert, stopwords, divide_by=1):
    """
    Compute document embeddings using PhoBERT in batch mode to optimize GPU usage.

    Args:
        segmentised_doc (list): List of segmented sentences.
        tokenizer: PhoBERT tokenizer.
        phobert: Pretrained PhoBERT model.
        stopwords (set): Set of stopwords to remove.
        divide_by (int): Factor to control batch size.

    Returns:
        torch.Tensor: 768-dimensional document embedding.
    """

    # Remove stopwords from each sentence
    cleaned_sentences = [
        " ".join([word for word in sentence.split() if word not in stopwords])
        for sentence in segmentised_doc
    ]

    # Determine batch size (ensuring at least 1 sentence per batch)
    batch_size = max(1, math.ceil(len(cleaned_sentences) / divide_by))

    # Initialize tensor for embeddings
    doc_embedding = torch.zeros(len(cleaned_sentences), 768).to(device)

    # Process sentences in batches
    for i in range(0, len(cleaned_sentences), batch_size):
        batch_sentences = cleaned_sentences[i : i + batch_size]  

        # Tokenize batch with padding/truncation
        encoded_inputs = tokenizer(
            batch_sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        # Get embeddings from PhoBERT
        with torch.no_grad():
            features = phobert(**encoded_inputs)

        # Store embeddings
        for k in range(features.pooler_output.size(0)):  
            index = min(i + k, len(doc_embedding) - 1)  # Prevent out-of-bounds indexing
            doc_embedding[index, :] = features.pooler_output[k] * (2 if i == 0 and k == 0 else 1)

    # Compute final document embedding (average of sentence embeddings)
    return torch.mean(doc_embedding, dim=0)

def get_segmentised_doc(nlp, rdrsegmenter, title, doc):
    for i, j in ethnicity_dict_map.items():
        if title is not None:
            title = title.replace(i, j)
        doc = doc.replace(i, j)

    segmentised_doc = rdrsegmenter.word_segment(doc)

    if title is not None:
        segmentised_doc = rdrsegmenter.word_segment(title) + rdrsegmenter.word_segment(doc)
    ne_ls = set(get_named_entities(nlp, doc))

    segmentised_doc_ne = []
    for sent in segmentised_doc:
        for ne in ne_ls:
            sent = sent.replace(ne, '_'.join(ne.split()))
        segmentised_doc_ne.append(sent)
    return ne_ls, segmentised_doc_ne


def compute_ngram_embeddings(tokenizer, phobert, ngram_list, divide_by=1):
    """
    Compute embeddings for multiple n-grams efficiently in batch mode.

    Args:
        tokenizer: PhoBERT tokenizer.
        phobert: Pretrained PhoBERT model.
        ngram_list (list): List of n-grams (phrases) to embed.
        divide_by (int): Factor to control batch size.

    Returns:
        dict: Dictionary of {ngram: embedding}.
    """

    # Normalize n-grams (convert uppercase n-grams to lowercase)
    cleaned_ngrams = [ngram.lower() if ngram.isupper() else ngram for ngram in ngram_list]

    # Determine batch size
    batch_size = max(1, math.ceil(len(cleaned_ngrams) / divide_by))

    # Initialize dictionary to store embeddings
    ngram_embeddings = {}

    # Process n-grams in batches
    for i in range(0, len(cleaned_ngrams), batch_size):
        batch_ngrams = cleaned_ngrams[i : i + batch_size]  # Get a batch of n-grams

        # Tokenize batch with padding/truncation
        encoded_inputs = tokenizer(
            batch_ngrams, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        # Get embeddings from PhoBERT
        with torch.no_grad():
            features = phobert(**encoded_inputs)

        # Store embeddings
        for k, ngram in enumerate(batch_ngrams):
            ngram_embeddings[ngram] = features.pooler_output[k]

    return ngram_embeddings


def compute_ngram_similarity(ngram_list, ngram_embeddings, doc_embedding):
    return {
        ngram: cosine_similarity(ngram_embeddings[ngram].squeeze(), doc_embedding.squeeze()).flatten()[0]
        for ngram in ngram_list
    }


def diversify_result_kmeans(ngram_result, ngram_embeddings, top_n=5):
    best_ngrams = sorted(ngram_result, key=ngram_result.get, reverse=True)[:top_n * 4]
    best_ngram_embeddings = np.array([ngram_embeddings[ngram] for ngram in best_ngrams]).squeeze()
    vote = {}

    for niter in range(100):
        kmeans = KMeans(n_clusters=top_n, init='k-means++', random_state=niter * 2, n_init="auto").fit(
            best_ngram_embeddings)
        kmeans_result = kmeans.labels_

        res = {}
        for i in range(len(kmeans_result)):
            if kmeans_result[i] not in res:
                res[kmeans_result[i]] = []
            res[kmeans_result[i]].append((best_ngrams[i], ngram_result[best_ngrams[i]]))

        final_result = [res[k][0] for k in res]
        for keyword in final_result:
            if keyword not in vote:
                vote[keyword] = 0
            vote[keyword] += 1

    diversify_result_ls = sorted(vote, key=vote.get, reverse=True)

    return diversify_result_ls[:top_n]


def remove_duplicates(ngram_result):
    to_remove = set()
    for ngram in ngram_result:

        for ngram2 in ngram_result:
            if ngram not in to_remove and ngram != ngram2 and ngram.lower() == ngram2.lower():
                new_score = np.mean([ngram_result[ngram], ngram_result[ngram2]])

                ngram_result[ngram] = new_score
                to_remove.add(ngram2)

    for ngram in to_remove:
        ngram_result.pop(ngram)
    return ngram_result


def compute_filtered_text(annotator, title, text):
    annotated = annotator.annotate_text(text)
    if title is not None:
        annotated = annotator.annotate_text(title + '. ' + text)
    filtered_sentences = []
    keep_tags = ['N', 'Np', 'V', 'Nc']
    for key in annotated.keys():
        # print(key,annotated[key])
        sent = ' '.join([dict_['wordForm'] for dict_ in annotated[key] if dict_['posTag'] in keep_tags])
        filtered_sentences.append(sent)
    return filtered_sentences


def get_candidate_ngrams(segmentised_doc, filtered_segmentised_doc, ngram_n, stopwords_ls):
    # get actual ngrams
    actual_ngram_list = compute_ngram_list(segmentised_doc, ngram_n, stopwords_ls, subsentences=True)

    # get filtered ngrams
    filtered_ngram_list = compute_ngram_list(filtered_segmentised_doc, ngram_n, stopwords_ls,
                                             subsentences=False)

    # get candiate ngrams
    candidate_ngram = [ngram for ngram in filtered_ngram_list if ngram in actual_ngram_list]
    return candidate_ngram


def limit_minimum_frequency(doc_segmentised, ngram_list, min_freq=1):
    ngram_dict_freq = {}
    for ngram in ngram_list:
        ngram_n = len(ngram.split())
        count = 0
        for sentence in doc_segmentised:
            sent = sentence.split()
            # print(sent)
            for i in range(len(sent) - ngram_n + 1):
                pair = ' '.join(sent[i:i + ngram_n])
                # print(pair, ngram)
                if pair == ngram:
                    count += 1
            # print(ngram, count)
        if count >= min_freq:
            ngram_dict_freq[ngram] = count

    return ngram_dict_freq


def remove_overlapping_ngrams(ngram_list):
    to_remove = set()
    for ngram1 in ngram_list:
        for ngram2 in ngram_list:
            if len(ngram1.split()) > len(ngram2.split()) and (ngram1.startswith(ngram2) or ngram1.endswith(ngram2)):
                # print(ngram1, ngram2)
                # print()
                to_remove.add(ngram2)

    # print("To removed")
    # print(to_remove)
    for kw in to_remove:
        ngram_list.remove(kw)
    return ngram_list

