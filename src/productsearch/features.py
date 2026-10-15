import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer

FEATURE_NAMES = ["bm25_title","bm25_all","tfidf_title","tfidf_all","coverage_title","coverage_all",
                 "phrase_title","phrase_all","coverage_brand","coverage_color","number_coverage","number_missing",
                 "title_length","document_length","query_length","brand_in_query","color_in_query"]


def tokens(text):
    return [token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if token not in ENGLISH_STOP_WORDS]


class BM25:
    def fit(self, texts, max_features=50000):
        self.vectorizer = CountVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False, max_features=max_features, dtype=np.float32)
        counts = self.vectorizer.fit_transform(texts).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        average = max(float(lengths.mean()),1)
        df = np.bincount(counts.indices,minlength=counts.shape[1])
        idf = np.log(1+(counts.shape[0]-df+0.5)/(df+0.5))
        rows = np.repeat(np.arange(counts.shape[0]),np.diff(counts.indptr))
        counts.data = (counts.data * 2.5 / (counts.data + 1.5*(0.25+0.75*lengths[rows]/average)) * idf[counts.indices]).astype(np.float32)
        self.matrix = counts
        self.inverted = counts.tocsc()
        return self

    def scores(self, query, ids=None):
        query_vector = self.vectorizer.transform([query])
        # Count a repeated query term once, matching the features used in training.
        query_vector.data[:] = 1
        if ids is not None:
            return (self.matrix[ids] @ query_vector.T).toarray().ravel()
        if not query_vector.nnz:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        return np.asarray(self.inverted[:,query_vector.indices].sum(axis=1)).ravel()


class FeatureExtractor:
    def fit(self, products, max_features=50000):
        self.products = products.reset_index(drop=True)
        self.titles = self.products.product_title.str.lower().tolist()
        self.documents = (self.products.product_title + " " + self.products.product_bullet_point + " " + self.products.product_description).str.lower().tolist()
        self.brands = self.products.product_brand.str.lower().tolist()
        self.colors = self.products.product_color.str.lower().tolist()
        self.title_tokens = [set(tokens(text)) for text in self.titles]
        self.document_tokens = [set(tokens(text)) for text in self.documents]
        self.title_bm25 = BM25().fit(self.titles,max_features)
        self.all_bm25 = BM25().fit(self.documents,max_features)
        self.tfidf = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False, max_features=max_features, dtype=np.float32)
        self.all_vectors = self.tfidf.fit_transform(self.documents)
        self.title_vectors = self.tfidf.transform(self.titles)
        return self

    def transform(self, query, ids):
        ids = np.asarray(ids,dtype=int)
        if not len(ids):
            return pd.DataFrame(columns=FEATURE_NAMES, dtype=np.float32)
        query_terms = set(tokens(query))
        n = max(len(query_terms),1)
        phrase = " ".join(tokens(query))
        qvector = self.tfidf.transform([query])
        columns = [self.title_bm25.scores(query,ids),self.all_bm25.scores(query,ids),
                   (self.title_vectors[ids] @ qvector.T).toarray().ravel(),(self.all_vectors[ids] @ qvector.T).toarray().ravel()]
        qnumbers = set(re.findall(r"\d+",query))
        extra=[]
        for row in ids:
            title, document = self.titles[row], self.documents[row]
            brand, color = self.brands[row], self.colors[row]
            numbers = set(re.findall(r"\d+",document))
            extra.append([
                len(query_terms & self.title_tokens[row])/n, len(query_terms & self.document_tokens[row])/n,
                float(bool(phrase) and phrase in " ".join(tokens(title))),float(bool(phrase) and phrase in " ".join(tokens(document))),
                len(query_terms & set(tokens(brand)))/n,len(query_terms & set(tokens(color)))/n,
                len(qnumbers & numbers)/max(len(qnumbers),1),float(bool(qnumbers - numbers)),
                np.log1p(len(self.title_tokens[row])),np.log1p(len(self.document_tokens[row])),float(len(query_terms)),
                float(bool(brand) and brand in query.lower()),float(bool(color) and color in query.lower()),
            ])
        result = np.column_stack(columns + [np.asarray(extra,dtype=np.float32)])
        return pd.DataFrame(result,columns=FEATURE_NAMES)

    def for_pairs(self, frame):
        result = pd.DataFrame(index=frame.index, columns=FEATURE_NAMES, dtype=np.float32)
        for _, group in frame.groupby("query_group",sort=False):
            result.loc[group.index] = self.transform(group.iloc[0]["query"],group.catalog_row.to_numpy()).to_numpy()
        return result.reset_index(drop=True)
