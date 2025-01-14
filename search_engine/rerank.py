from transformers import AutoTokenizer, AutoModel
import torch
from pymongo import MongoClient
import torch.nn.functional as F
import os

# Set environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
model = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
# Kết nối đến MongoDB một lần
uri = "mongodb+srv://admin1:vinh1950@chatbot1.r8ahn.mongodb.net/"
client = MongoClient(uri)
db = client['items1']
collection = db['items_with_embedding']

def get_embedding(item):
    inputs = tokenizer(item, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  
    
    return embeddings.squeeze()  


class ReRanker():
    def __init__(self):
        pass

    def rank(self, q:str, top_k:list):
        embedding_q = get_embedding(q)
        # print(embedding_q.shape)
        similarities = []
        
        for doc in top_k:
            title = doc[1]
            document = collection.find_one({'text': title})
            if document is not None:
                embedding = torch.tensor(document['embedding'])  
                # Tính cosine similarity
                cos = torch.cosine_similarity(embedding.unsqueeze(0), embedding_q.unsqueeze(0))
                similarities.append((title, cos))  
                # print(embedding.shape)
  

        # Sắp xếp các tài liệu theo cosine similarity
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        return sorted_similarities
                


if __name__ == '__main__':
    rerank = ReRanker()
    top_k = ['Điện thoại Honor X5b (4+64GB)', 'HONOR X7c - 8GB/256GB']
    print(rerank.rank(q='điện thoại Honnor', top_k=top_k))