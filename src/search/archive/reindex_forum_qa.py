"""
Re-index forum Q&A pairs in Elasticsearch with Phase 2 improvements.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from elasticsearch import Elasticsearch

# Import forum_indexer functions
import importlib.util
spec = importlib.util.spec_from_file_location("forum_indexer", project_root / "src" / "search" / "forum_indexer.py")
forum_indexer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forum_indexer)

def main():
    # Connect to Elasticsearch
    es = Elasticsearch(['http://localhost:9200'])
    
    print("Step 1: Deleting old forum documents...")
    response = es.delete_by_query(
        index='rulebook_chunks',
        body={'query': {'term': {'source_type': 'forum'}}}
    )
    print(f"Deleted {response['deleted']} forum documents")
    
    # Force refresh to ensure deletion is visible
    es.indices.refresh(index='rulebook_chunks')
    
    print("\nStep 2: Re-indexing with Phase 1+2 improvements...")
    
    # Call main from forum_indexer (with skip rulebook update since we're only updating forum docs)
    forum_indexer.main(
        qa_json_path=str(project_root / 'data' / 'processed' / 'forum_qa' / 'forum_qa_pairs.json'),
        es_host='http://localhost:9200',
        update_rulebook=False  # Skip since we're only re-indexing forum
    )
    
    # Get final counts
    print("\nStep 3: Verifying counts...")
    es.indices.refresh(index='rulebook_chunks')
    
    total_docs = es.count(index='rulebook_chunks')['count']
    forum_docs = es.count(index='rulebook_chunks', body={'query': {'term': {'source_type': 'forum'}}})['count']
    rulebook_docs = es.count(index='rulebook_chunks', body={'query': {'term': {'source_type': 'rulebook'}}})['count']
    
    print(f"\nFinal document counts:")
    print(f"  Total: {total_docs}")
    print(f"  Rulebook: {rulebook_docs}")
    print(f"  Forum: {forum_docs}")
    
    print("\n✓ Re-indexing complete!")

if __name__ == "__main__":
    main()
