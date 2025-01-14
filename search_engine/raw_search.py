from collections import Counter
import re
import math
import numpy as np
from pymongo import MongoClient

def nomalize_sentence(doc):
    doc = doc.lower()
    words = re.findall(r'[a-zA-Z0-9àáạảãâầấậẩẫèéẹẻẽêềếệểễđìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ]+', doc)
    return words

class BM25():
    def __init__(self):
        self.k1 = 1.25
        self.b = 0.75
        self.tf = {}
        self.idf = {}

    def load_data(self, docs:list, use_data=True):
        self.title = []
        if use_data:
            self.title = [doc[1] for doc in docs]
            docs = [doc[0] for doc in docs]
        self.docs = docs
        self.words_docs = [nomalize_sentence(doc) for doc in docs]
        self.l = len(self.docs)
        self.l_docs = [len(doc) for doc in self.words_docs]
        self.avg_doc_length = sum(self.l_docs) / self.l

    def build_vocab(self):
        vocab = set()
        for doc in self.docs:
            vocab.update(doc)
        self.freq = {word: [0] * self.l for word in vocab}
        self.tf = {word: [0] * self.l for word in vocab}
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
                self.idf[word] = math.log10(self.l / df)

    def load_data_by_using_db(self):
        uri = "mongodb+srv://admin1:vinh1950@chatbot1.r8ahn.mongodb.net/"
        client = MongoClient(uri)
        db = client['items1']
        collection1 = db['tf_idf']
        collection2 = db['items_without_embedding']
        tf_idfs = collection1.find({})
        for tf_idf in tf_idfs:
            word = tf_idf['word']
            self.tf[word] = tf_idf['tf']
            self.idf[word] = tf_idf['idf']
        items = collection2.find()
        docs = []
        for item in items:
            docs.append((item['full'], item['title']))
        self.load_data(docs)

    def search(self, q: str, k: int, use_db=True):

            
        query_words = nomalize_sentence(q)
        scores = {}
        for i in range(self.l):
            score = 0.0
            for word in query_words:
                if word in self.tf:
                    tf = self.tf[word][i]
                    idf = self.idf[word]
                    score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (self.l_docs[i] / self.avg_doc_length)))
            scores[i] = score
        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        if use_db:
            return [(self.docs[i], self.title[i]) for i, score in ranked_docs]
        return [(self.docs[i], score) for i, score in ranked_docs]

if __name__ == "__main__":
    bm25 = BM25()
    bm25.load_data_by_using_db()
    print(bm25.search(q='samsung s24', k=10))
    
    # # Tải dữ liệu vào BM25
    # bm25.load_data(docs)
    
    # # Xây dựng từ vựng
    # bm25.build_vocab()
    
    # # Tính toán TF
    # bm25.cal_tf()
    
    # # Tính toán IDF
    # mnp = bm25.cal_idf()
    # print(mnp)
    
    # # Thực hiện tìm kiếm với một truy vấn
    # query = "học sâu"
    # k = 3  # Số lượng tài liệu trả về
    # results = bm25.search(query, k)
    
    # # In kết quả tìm kiếm
    # for doc, score in results:
    #     print(f"Document: {' '.join(doc)}, Score: {score}")
    # print(bm25.freq)
