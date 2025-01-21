from collections import Counter
import re
import math
import numpy as np
from pymongo import MongoClient

def nomalize_sentence(doc):
    #input: str
    #output: list word
    doc = doc.lower()
    words = re.findall(r'[a-zA-Z0-9àáạảãâầấậẩẫèéẹẻẽêềếệểễđìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ]+', doc)
    return words

class BM25():
    def __init__(self):
        self.k1 = 1.25
        self.b = 0.75
        self.docs = None
        self.tf = {}
        self.idf = {}

    def load_data(self, docs:list, use_data=True):
        self.index_in_tf_idf = []
        if use_data:
            self.index_in_tf_idf = [doc[0] for doc in docs]
            self.docs = [doc[1] for doc in docs]

        self.num_doc = len(self.docs)
        if not use_data:
            self.words_docs = [nomalize_sentence(doc) for doc in docs]
            self.docs_length = [len(doc) for doc in self.words_docs]
            
        self.avg_doc_length = sum(self.docs_length) / self.num_doc
        
    def load_data_by_using_db(self):
        #connect to db
        uri = "mongodb://localhost:27017"
        client = MongoClient(uri)
        db = client['items']
        collection1 = db['tf_idf']
        collection2 = db['items_without_embedding']
        tf_idfs = collection1.find()
        for tf_idf in tf_idfs:
            word = tf_idf['word']
            self.tf[word] = tf_idf['tf']
            self.idf[word] = tf_idf['idf']
        items = collection2.find()
        docs = []
        self.docs_length = []
        for item in items:
            docs.append((item['index_in_tf_idf'], item['full']))
            self.docs_length.append(item['len_doc'])
        self.load_data(docs)

    def build_vocab(self):
        vocab = set()
        for doc in self.docs:
            vocab.update(doc)
        self.freq = {word: [0] * self.num_doc for word in vocab}
        self.tf = {word: [0] * self.num_doc for word in vocab}
        self.idf = {word: 0 for word in vocab}

    def cal_tf(self):
        for i, doc in enumerate(self.docs):
            counter = Counter(doc)
            for word, count in counter.items():
                self.freq[word][i] = count
                self.tf[word][i] = 1 + math.log10(count)

    def cal_idf(self):
        for word in self.idf.keys():
            df = sum(1 for freq in self.freq[word] if freq > 0)
            if df > 0:
                self.idf[word] = math.log10(self.num_doc / df)

    

    def search(self, q: str, k: int, use_db=True):
        query_words = nomalize_sentence(q)
        scores = {}
        for i in range(self.num_doc):
            score = 0.0
            for word in query_words:
                if word in self.tf:
                    tf = self.tf[word][i]
                    idf = self.idf[word]
                    score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (self.docs_length[i] / self.avg_doc_length)))
            scores[i] = score
            
        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        if use_db:
            return [(self.docs[i], self.index_in_tf_idf[i]) for i, score in ranked_docs]
        return [(self.docs[i], score) for i, score in ranked_docs]

if __name__ == "__main__":
    bm25 = BM25()
    bm25.load_data_by_using_db()
    print(bm25.search(q='samsung s24', k=10))
    
