#input: query 
#output: is need to use RAG(0 or 1)
#0: not need to use RAG
#1: need to use RAG


class Router():
    def __init__(self):
        pass
    
    def redict(self, filtered_results):
        max_similarity = filtered_results[0][1]
        
        
        return 1 if max_similarity > 0.26 else 0
    
    
if __name__ == '__main__':
    pass