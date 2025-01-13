#input: query 
#output: is need to use RAG(0 or 1)
import random

class Router():
    def __init__(self):
        pass
    
    def redict(self, q: str):
        
        
        return random.randint(0,1)
    
    
if __name__ == '__main__':
    r = Router()
    print(r.redict(''))