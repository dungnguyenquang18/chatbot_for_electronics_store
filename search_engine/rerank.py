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
        # connect to db
        uri = "mongodb://localhost:27017"
        self.client = MongoClient(uri)
        self.db = self.client['items']
        self.collection = self.db['items_with_embedding']
        self.index = None  #faiss

    def get_embedding(self, item):
        inputs = self.tokenizer(item, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings #shape: [1, 768]

    def build_index(self, embeddings):
        #faiss using inner product(cosine simalarity because vector in db is normed)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
    
    def norm_vector(self, tensor:torch.Tensor):
        return tensor / tensor.norm(dim=1, keepdim=True)
    
    def rank(self, q: str, top_k: list, limit: int):
        embedding_q = self.norm_vector(self.get_embedding(q))  #shape: [1, 768]
        #get embeddings in db by index_in_tf_idf and stack them
        embeddings = []
        
        for doc in top_k:
            indx = doc[1]
            document = self.collection.find_one({'index_in_tf_idf': indx})
            if document is not None:
                embeddings.append(torch.tensor(document['full_embed']))  

        embeddings = torch.stack(embeddings).numpy()  #shape: [num_doc, 1, 768]

        #build faiss
        if self.index is None:
            self.build_index(embeddings.squeeze(axis=1)) 

        # rerank 
        distances, indices = self.index.search(embedding_q.numpy(), limit)  
        similarities = []
        for i in range(limit):
            cos_similarity = distances[0][i]  
            similarities.append((top_k[indices[0][i]][0], float(cos_similarity)))

        return similarities[:limit]

if __name__ == '__main__':
    from raw_search import BM25
    rerank = ReRanker()
    bm25 = BM25()
    bm25.load_data_by_using_db()
    t = int(input())
    for _ in range(t):
        q = input()
        k = 20
        top_k = bm25.search(q, k)
        new_top_k = rerank.rank(q, top_k, k // 2)
        print(new_top_k)