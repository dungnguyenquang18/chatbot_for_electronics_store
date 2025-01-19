from flask import Flask, request, jsonify
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display, Markdown
from rerank import ReRanker
from raw_search import BM25

app = Flask(__name__)
bm25 = BM25()

def to_markdown(text:str):
    text = text.replace('.',' *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _:True))

def prompt(q:str, scores:list):
    new_query = q + "biết rằng:\n"
    for infor in scores:
        new_query += str(infor[0]) + "\n"
    
    return new_query
    
@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    #load data from db
    bm25.load_data_by_using_db(field='title')

    # Example response based on the query
    clean_query = process_query(query)

    # Search using TF-IDF
    filtered_results = bm25.search(clean_query, 10)

    # return jsonify({'res': filtered_results})

    # Re-rank using AI 
    re_ranker = ReRanker()

    scores = re_ranker.rank(query, filtered_results, 10, field='full')
        
    genai.configure(api_key="AIzaSyCpLV3fieMeLX_IrlQGq17mesLZgKeQ1Ho")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt(clean_query, scores))


    return jsonify(response.text)
    # return jsonify({'res': filtered_results})

# AIzaSyCpLV3fieMeLX_IrlQGq17mesLZgKeQ1Ho

def process_query(query):
    return query

if __name__ == '__main__':
    app.run(debug=True)
