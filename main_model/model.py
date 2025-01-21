from router import Router
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display, Markdown

class Model():
    def __init__(self):
        #using gemini
        genai.configure(api_key="AIzaSyCpLV3fieMeLX_IrlQGq17mesLZgKeQ1Ho")
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
    def reprompt(self, q, informations):
        new_query = q + "biết rằng:\n"
        for infor in informations:
            new_query += str(infor[0]) + "\n"
        
        return new_query
    
    def answer(self, q, informations):
        prompt = q
        router = Router()
        if router.redict(informations) == 1:
            #reprompt
            prompt = self.reprompt(q, informations)
            
        response = self.model.generate_content(prompt)    
        
        return response.text
        