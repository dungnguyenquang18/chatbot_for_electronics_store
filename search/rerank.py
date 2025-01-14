from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

class Rerank():
    
    def __init__(self):
        pass
    
    def sort(self, q:str, top_k: dict):
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        model = AutoModelForMaskedLM.from_pretrained('vinai/phobert-base')
        input_q = tokenizer(q, return_tensors="pt", truncation=True, max_length=512, padding=True)  # Xử lý batch
        output_q = model(**input)
        