from flask import Flask, request, jsonify
from main_model import Model
from search_engine import ReRanker, BM25


app = Flask(__name__)
bm25 = BM25()
re_ranker = ReRanker()
model = Model()
bm25.load_data_by_using_db()


@app.route('/api/search', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    #raw_search
    raw_search_informations = bm25.search(q=query, k=35)
    #rerank 
    informations = re_ranker.rank(q=query, top_k=raw_search_informations, limit=10)
    #get answer by reprompting
    answer = model.answer(query, informations)
    
    return jsonify(answer)


if __name__ == '__main__':
    app.run(debug=True)
