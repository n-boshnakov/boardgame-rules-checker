"""
Generate a detailed markdown report of all QA results with retrieval information
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from src.search.retriever import RulebookRetriever

# Load questions
df = pd.read_csv('data/processed/qa_results.csv')
print(f"Loaded {len(df)} questions")

# Initialize retriever
print("Initializing retriever...")
retriever = RulebookRetriever()

# Generate report
output_lines = []
output_lines.append("# Board Game Rulebook Q&A Report\n")
output_lines.append(f"Generated answers for {len(df)} questions using hybrid search with cross-encoder re-ranking.\n")
output_lines.append("---\n")

for idx, row in df.iterrows():
    question = row['question']
    ground_truth = row['ground_truth']
    
    print(f"Processing question {idx+1}/{len(df)}: {question[:50]}...")
    
    # Get top 5 results
    results = retriever.search(question, top_k=5)
    
    output_lines.append(f"\n## {idx+1}. {question}\n")
    output_lines.append(f"**Ground Truth:** {ground_truth}\n")
    output_lines.append("")
    
    for rank, result in enumerate(results, 1):
        score = result.get('score', 0)
        section = result.get('section', 'N/A')
        page = result.get('page', 'N/A')
        text = result.get('text', '')
        chunk_index = result.get('chunk_index', 'N/A')
        
        # Add cross-encoder score if available
        ce_score = result.get('cross_encoder_score', None)
        score_display = f"{score:.3f}"
        if ce_score is not None:
            score_display += f" (CE: {ce_score:.3f})"
        
        output_lines.append(f"### Answer {rank}: {section} - Score: {score_display}\n")
        output_lines.append(f"**Metadata:** Page {page}, Section: {section}, Chunk: {chunk_index}\n")
        output_lines.append(f"{text}\n")
        output_lines.append("")

# Write to file
output_path = 'data/processed/qa_retrieval_report.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\nReport saved to {output_path}")
print(f"Total lines: {len(output_lines)}")
