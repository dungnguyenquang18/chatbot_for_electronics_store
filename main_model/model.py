from router import Router

class Model():
    def __init__(self):
        pass
    
    def reprompt(self, information):
        return ''
        pass
    
    def answer(self, q):
        prompt = q
        router = Router()
        if router.redict(q) == 1:
            #cal information: 
            information = []
            #reprompt
            prompt = self.reprompt(information)
            
        #prompt       
        pass