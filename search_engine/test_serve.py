from flask import Flask, request, jsonify

app = Flask(__name__)

from rerank import ReRanker
from raw_search import BM25

bm25 = BM25()


@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    #load data from db
    bm25.load_data_by_using_db(field='title')

    # Example response based on the queryi
    clean_query = process_query(query)

    # Search using TF-IDF
    filtered_results = bm25.search(clean_query, 20)

    # Re-rank using AI 
    re_ranker = ReRanker()

    scores = re_ranker.rank(query, filtered_results, 10, field='title')

    return jsonify({'response': scores})

def process_query(query):
    return query

if __name__ == '__main__':
    app.run(debug=True)
