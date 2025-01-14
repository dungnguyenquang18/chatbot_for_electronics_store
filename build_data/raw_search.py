from collections import Counter
import re
import math
import numpy as np


def nomalize_sentence(doc):
    #input: a document
    #output: a list of nomalized word
    doc = doc.lower()
    words = re.findall(r'[a-zA-Z0-9àáạảãâầấậẩẫèéẹẻẽêềếệểễđìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ]+', doc)
    return words

class BM25():
    def __init__(self):
        self.k1 = 1.25
        self.b = 0.75
        
    
    def load_data(self, docs:dict):
        self.docs = [nomalize_sentence(doc) for doc in docs]
        self.l = len(self.docs)
        self.avg_doc_length = np.sum(np.array([len(doc) for doc in self.docs]))/self.l
 
        
    def build_vocab(self):

        vocab = set()
        for doc in self.docs:
            vocab.update(doc)  # Cập nhật từ vựng từ danh sách từ
        

        self.freq = {word: [0] * self.l for word in vocab}  # Tạo từ điển freq
        self.tf = {word: [0] * self.l for word in vocab}  # Tạo từ điển tf
        self.idf = {word: 0 for word in vocab}
        # self.tf_idf = {word: [0] * self.l for word in vocab}
        

    def cal_tf(self):
        for i, doc in enumerate(self.docs):
            counter = Counter(doc)
            for word, count in counter.items():
                self.freq[word][i] = count
                self.tf[word][i] = 1 + math.log10(count) 

        
    
                
    def cal_idf(self):
        for word in self.idf.keys():
            df = 0
            for freq in self.freq[word]:
                if freq > 0:
                    df += 1
            self.idf[word] = math.log10(self.l/df)
        # return self.freq, self.idf
        

    def search(self, q: str, k: int):
        # Tiền xử lý truy vấn
        query_words = nomalize_sentence(q)
        scores = {}

        # Tính toán điểm BM25 cho mỗi tài liệu
        for i, doc in enumerate(self.docs):
            score = 0.0
            doc_length = len(doc)
            
            for word in query_words:
                if word in self.freq.keys():
                    # Tính toán BM25 cho từ hiện tại
                    tf = self.tf[word][i]
                    idf = self.idf[word]
                    score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length)))

            scores[i] = score

        # Sắp xếp các tài liệu theo điểm số và lấy k tài liệu có điểm cao nhất
        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        
        # Trả về các tài liệu có điểm cao nhất
        return [(self.docs[i], score) for i, score in ranked_docs]

if __name__ == "__main__":
    # Tạo danh sách tài liệu mẫu
    docs = [
        "Học máy là một lĩnh vực thú vị.",
        "Học sâu là một nhánh của học máy.",
        "Học máy có nhiều ứng dụng trong thực tế.",
        "Học máy và học sâu đều sử dụng dữ liệu.",
        "Dữ liệu lớn là một thách thức trong học máy."
    ]

    # Khởi tạo đối tượng BM25
    bm25 = BM25()
    
    # Tải dữ liệu vào BM25
    bm25.load_data(docs)
    
    # Xây dựng từ vựng
    bm25.build_vocab()
    
    # Tính toán TF
    bm25.cal_tf()
    
    # Tính toán IDF
    mnp = bm25.cal_idf()
    print(mnp)
    
    # Thực hiện tìm kiếm với một truy vấn
    query = "học sâu"
    k = 3  # Số lượng tài liệu trả về
    results = bm25.search(query, k)
    
    # In kết quả tìm kiếm
    for doc, score in results:
        print(f"Document: {' '.join(doc)}, Score: {score}")
    print(bm25.freq)
