from transformers import AutoTokenizer, AutoModel
import torch
from pymongo import MongoClient
import numpy as np
import faiss
import os

# Set environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ReRanker():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("keepitreal/vietnamese-sbert")
        self.model = AutoModel.from_pretrained("keepitreal/vietnamese-sbert")
        # Kết nối đến MongoDB một lần
        uri = "mongodb://localhost:27017"
        self.client = MongoClient(uri)
        self.db = self.client['items']
        self.collection = self.db['items_with_embedding']
        self.index = None  # Khởi tạo chỉ mục FAISS ở đây

    def get_embedding(self, item):
        inputs = self.tokenizer(item, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def build_index(self, embeddings):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
    
    def norm_vector(self, tensor:torch.Tensor):
        return tensor/tensor.norm(dim=1, keepdim=True)
    
    def rank(self, q: str, top_k: list, limit: int, field: str):
        embedding_q = self.norm_vector(self.get_embedding(q))  # Giữ nó ở dạng tensor
        embeddings = []
        # Lấy embedding từ cơ sở dữ liệu cho các tài liệu trong top_k
        for doc in top_k:
            indx = doc[1]
            document = self.collection.find_one({'index_in_tf_idf': indx})
            if document is not None:
                embeddings.append(torch.tensor(document[field + '_embed']))  # Giữ nó ở dạng tensor


        # Chuyển đổi danh sách embeddings thành mảng tensor
        embeddings = torch.stack(embeddings).numpy()  # Chuyển đổi thành numpy để FAISS
        # print(embeddings.squeeze(axis=1).shape)
        # Xây dựng chỉ mục FAISS
        if self.index is None:
            self.build_index(embeddings.squeeze(axis=1))

        # Sử dụng FAISS để tìm kiếm các tài liệu tương tự
        distances, indices = self.index.search(embedding_q.numpy(), limit)  # Tìm kiếm k gần nhất
        similarities = []
        for i in range(limit):
            cos_similarity = distances[0][i]  
            similarities.append((top_k[indices[0][i]][2], float(cos_similarity)))

        return similarities[:limit]

if __name__ == '__main__':
    from raw_search import BM25
    rerank = ReRanker()
    bm25 = BM25()
    bm25.load_data_by_using_db(field='title')
    t = int(input())
    for _ in range(t):
        q = input()
        k = 20
        top_k = bm25.search(q, k)
        new_top_k = rerank.rank(q, top_k, k // 2, 'full')
        print(new_top_k)