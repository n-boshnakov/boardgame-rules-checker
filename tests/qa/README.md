# QA Testing Utilities

This directory contains testing and validation scripts for the Question Answering system.

## Testing Scripts

### Quick Validation

**`quick_check.py`** - Fast validation of QA results
```bash
python tests/qa/quick_check.py
```
- Validates current `qa_results.csv`
- Reports: correct count, success rate, mean/median similarity
- Breaks down first 20 questions separately
- Use for rapid iteration and debugging

### Retriever Accuracy Test

**`test_retriever_accuracy.py`** - Compare retriever vs batch results
```bash
python tests/qa/test_retriever_accuracy.py
```
- Tests first 10 questions
- Compares batch script vs interactive retriever
- Shows similarity differences per question
- Use to ensure retriever matches batch accuracy after changes

### Full Accuracy Test

**`test_full_accuracy.py`** - Comprehensive validation on all questions
```bash
python tests/qa/test_full_accuracy.py
```
- Tests all 40 questions
- Detailed comparison: batch vs retriever
- Identifies improved/declined questions
- Saves results to `data/processed/retriever_vs_batch_comparison.csv`
- Use before deploying changes to production

### Compare Results

**`compare_results.py`** - Compare two QA result files
```bash
python tests/qa/compare_results.py
```
- Shows questions with >5 point changes between runs
- Reports answer length changes
- Use to analyze impact of specific changes

## Typical Testing Workflow

### 1. Make Changes to Retriever
Edit `src/search/retriever.py` to modify answer generation logic

### 2. Quick Interactive Test
```bash
python src/search/retriever.py "Your test question here"
```

### 3. Validate on Sample
```bash
python tests/qa/test_retriever_accuracy.py
```
Expected: Within 2 points of batch accuracy

### 4. Full Validation
```bash
python tests/qa/test_full_accuracy.py
```
Expected: Overall similarity maintained or improved

### 5. Production Run
```bash
python scripts/run_qa_batch.py --max_questions 40
```

### 6. Quick Check Results
```bash
python tests/qa/quick_check.py
```

## Performance Benchmarks

Current performance (as of Dec 24, 2025):
- **Overall (40 questions):** 49.87% mean similarity
- **First 20 (answerable):** 59.76% mean similarity
- **≥70% threshold:** 8/40 questions
- **≥80% threshold:** 6/40 questions

## Key Insights

- Retriever and batch now produce identical results
- Processing 10 chunks (up from 5) handles ES ranking variability
- Max 10 sentences ensures comprehensive answers
- Q7 "Can anomalies move?" is the litmus test (should be 90.3%)
