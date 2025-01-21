from flask import Flask, request, jsonify
from main_model import Model
from search_engine import ReRanker, BM25


app = Flask(__name__)
bm25 = BM25()
re_ranker = ReRanker()
model = Model()
bm25.load_data_by_using_db(field='title')


@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    clean_query = process_query(query)

    filtered_results = bm25.search(q=clean_query, k=35)
    
    informations = re_ranker.rank(query, filtered_results, limit=10, field='full')
    
    answer = model.answer(clean_query, informations)
    
    return jsonify(answer)


def process_query(query:str):
    return query.lower()

if __name__ == '__main__':
    app.run(debug=True)
