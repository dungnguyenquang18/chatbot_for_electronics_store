#input: query 
#output: is need to use RAG(0 or 1)


class Router():
    def __init__(self):
        pass
    
    def redict(self, filtered_results):
        max_similarity = filtered_results[0][1]
        
        
        return 1 if max_similarity > 0.26 else 0
    
    
if __name__ == '__main__':
    r = Router()
    print(r.redict(''))