"""
Example script to create elasticsearch documents.
"""
import argparse
import codecs, json
import numpy as np
#import gluonnlp as nlp
#import mxnet as mx


import pandas as pd

from sentence_transformers import SentenceTransformer

##model, vocab = nlp.model.get_model('roberta_12_768_12', dataset_name='openwebtext_ccnews_stories_books_cased', use_decoder=False);

#tokenizer = nlp.data.GPT2BPETokenizer();


model  = "roberta-base-nli-stsb-mean-tokens"
embedder = SentenceTransformer(model)

def create_document(doc, emb, index_name):
    return {
        '_op_type': 'index',
        '_index': index_name,
        'asin': doc['asin'],
        'type': doc['image'],
        'type': doc['url'],
        'type': doc['type'],
        'title': doc['title'],
        'text': doc['text'],
        'text_vector': emb
    }


def load_dataset(path):
    docs = []
    df = pd.read_csv(path)
    for row in df.iterrows():
        series = row[1]
        doc = {
            'asin': series.Asin,
            'image': series.Img_URL,
            'url': series.Web_URL,
            'type': series.Type,
            'title': series.Title,
            'text': series.Text
        }
        docs.append(doc)
    return docs


def bulk_predict(docs, batch_size=256):
    """Predict bert embeddings."""
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
#        for d in batch_docs:
        embeddings= embedder.encode([doc['text'] for doc in batch_docs])
#           embeddings_bert = model(mx.nd.array([vocab[[vocab.bos_token] + tokenizer(d['text']) + [vocab.eos_token]]]))
#           embeddings =  embeddings_bert[:,0,:].flatten()
        for emb in embeddings:
           yield emb.tolist()
#.          yield embeddings[0].asnumpy().tolist()
            
def main(args):
    docs = load_dataset(args.data)
    with open(args.save, 'w') as f:
        for doc, emb in zip(docs, bulk_predict(docs)):
            d = create_document(doc, emb, args.index_name)
            f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch documents.')
    parser.add_argument('--data', help='data for creating documents.')
    parser.add_argument('--save', default='documents.jsonl', help='created documents.')
    parser.add_argument('--index_name', default='jobsearch', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)