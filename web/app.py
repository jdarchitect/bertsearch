import os
from pprint import pprint
from flask import Flask, render_template, jsonify, request
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import codecs, json
import numpy as np
import gluonnlp as nlp 
import mxnet as mx

SEARCH_SIZE = 10
INDEX_NAME = os.environ['INDEX_NAME']
#model  = "roberta-base-nli-stsb-mean-tokens"
#embedder = SentenceTransformer(model)
model, vocab = nlp.model.get_model('roberta_12_768_12', dataset_name='openwebtext_ccnews_stories_books_cased', use_decoder=False);
tokenizer = nlp.data.GPT2BPETokenizer();

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def analyzer():
    client = Elasticsearch('elasticsearch:9200')
    
    query = request.args.get('q')
    embeddings = model(mx.nd.array([vocab[[vocab.bos_token] + tokenizer(query) + [vocab.eos_token]]]))
    query_vector =  embeddings[:,0,:].flatten()[0]
#    query_vector= embedder.encode([query])[0]

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['text_vector'])+1.0",
                "params": {"query_vector": query_vector.asnumpy().tolist()}
            }
        }
    }

    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title", "text"]}
        }
    )
    print(query)
    pprint(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
