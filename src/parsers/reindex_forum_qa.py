"""
Re-index forum Q&A pairs.
Removes old forum documents and indexes new cleaned data.
"""

from elasticsearch import Elasticsearch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from search.forum_indexer import main as forum_indexer_main

ES_INDEX = "rulebook_chunks"


def delete_forum_documents(es_host: str = "http://localhost:9200"):
    """Delete all forum documents from the index."""
    print("="*80)
    print("DELETING OLD FORUM DOCUMENTS")
    print("="*80)
    
    es = Elasticsearch(es_host)
    
    # Check if index exists
    if not es.indices.exists(index=ES_INDEX):
        print(f"Index '{ES_INDEX}' does not exist. Nothing to delete.")
        return
    
    # Count current forum documents
    try:
        forum_stats = es.count(index=ES_INDEX, body={"query": {"term": {"source_type": "forum"}}})
        old_forum_count = forum_stats['count']
        print(f"\nFound {old_forum_count} old forum documents")
    except Exception as e:
        print(f"Error counting forum documents: {e}")
        old_forum_count = 0
    
    if old_forum_count == 0:
        print("No forum documents to delete.")
        return
    
    # Delete all forum documents
    print(f"Deleting {old_forum_count} forum documents...")
    
    try:
        result = es.delete_by_query(
            index=ES_INDEX,
            body={
                "query": {
                    "term": {
                        "source_type": "forum"
                    }
                }
            }
        )
        
        deleted_count = result.get('deleted', 0)
        print(f"✓ Deleted {deleted_count} forum documents")
        
        # Verify deletion
        forum_stats_after = es.count(index=ES_INDEX, body={"query": {"term": {"source_type": "forum"}}})
        remaining = forum_stats_after['count']
        
        if remaining == 0:
            print("✓ All forum documents successfully removed")
        else:
            print(f"⚠ Warning: {remaining} forum documents still remain")
        
    except Exception as e:
        print(f"Error deleting forum documents: {e}")
        raise


def main():
    """Main re-indexing workflow."""
    print("\n" + "="*80)
    print("FORUM Q&A RE-INDEXING")
    print("="*80)
    print("\nThis script will:")
    print("  1. Delete old forum documents from Elasticsearch")
    print("  2. Index new forum Q&A pairs")
    print("  3. Verify index statistics")
    print("\n" + "="*80)
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    qa_file = project_root / "data" / "processed" / "forum_qa" / "forum_qa_pairs.json"
    
    if not qa_file.exists():
        print(f"\nError: forum_qa_pairs.json not found at {qa_file}")
        print("Please run reprocess_forum_qa.py first.")
        sys.exit(1)
    
    print(f"\nUsing Q&A file: {qa_file}")
    
    # Step 1: Delete old forum documents
    es_host = "http://localhost:9200"
    delete_forum_documents(es_host)
    
    # Step 2: Index new forum Q&A pairs
    print("\n" + "="*80)
    print("INDEXING NEW FORUM Q&A PAIRS")
    print("="*80 + "\n")
    
    # Run forum indexer (skip rulebook update since we're just replacing forum docs)
    forum_indexer_main(
        qa_json_path=str(qa_file),
        es_host=es_host,
        update_rulebook=False  # Don't update rulebook docs again
    )
    
    print("\n" + "="*80)
    print("✓ RE-INDEXING COMPLETE")
    print("="*80)
    print("\nNew forum Q&A pairs are now indexed!")
    print("Ready for dual-source search testing.")


if __name__ == "__main__":
    main()
