from elasticsearch import Elasticsearch

es = Elasticsearch(['http://localhost:9200'])
indices = es.cat.indices(format='json')

print('Available indices:')
for idx in indices:
    if not idx['index'].startswith('.'):
        print(f"  - {idx['index']} ({idx['docs.count']} docs)")
