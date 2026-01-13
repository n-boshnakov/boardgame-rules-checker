# Boardgame Rules Checker - Web UI

A simple web interface for asking questions about board game rules and receiving answers powered by hybrid search and multi-dimensional scoring.

## Features

- **Question Input**: Ask natural language questions about game rules
- **Answer Display**: Clear answer text with source information
- **Quality Scores**: Multi-dimensional scoring breakdown (relevance, completeness, accuracy, conciseness)
- **Debug Information**: Expandable section showing:
  - Processing time
  - Number of chunks retrieved
  - Semantic analysis status
  - Top 5 chunks with scores and text
- **Source Attribution**: Shows which source (rulebook, FAQ, etc.) and page number
- **Confidence Indicator**: Visual confidence rating for the answer
- **Semantic Analysis Toggle**: Optional NLP enhancement

## Installation

### Prerequisites
- Python 3.11+
- Elasticsearch running on localhost:9200
- All dependencies from `requirements.txt` installed
- Indexed rulebook chunks in Elasticsearch

### Install Flask
```bash
pip install flask
```

## Running the UI

### Start the Web Server
```bash
cd src/ui
python app.py
```

The server will start at: **http://localhost:5000**

### Using the Interface

1. **View Game Info**: Current game and available sources are shown at the top
2. **Ask a Question**: Type your question in the text area
3. **Optional Settings**: 
   - Check "Use Semantic Analysis" for NLTK-based query enhancement
4. **Submit**: Click "Ask Question" or press Ctrl+Enter
5. **View Answer**: Answer appears with:
   - Answer text
   - Source type and page number
   - Confidence rating
   - Quality scores breakdown
6. **Debug Info**: Click "🔍 Debug Information" to see:
   - Processing time
   - Number of chunks retrieved
   - Top 5 chunks with scores and text

## Architecture

### Backend (`app.py`)
- Flask application serving REST API
- Initializes `RulebookRetriever` and `MultiDimensionalScorer`
- Handles `/ask` POST endpoint for question answering
- Returns structured JSON with answer and metadata

### Frontend (`templates/index.html`)
- Single-page application
- Form for question input
- Dynamic answer display
- Expandable debug section

### Styling (`static/style.css`)
- Clean, modern design
- Responsive layout
- Color-coded confidence and scores
- Smooth animations

### JavaScript (`static/script.js`)
- Handles form submission
- AJAX calls to backend API
- Dynamic DOM manipulation for results
- Error handling

## API Endpoints

### POST `/ask`
Submit a question and get an answer.

**Request:**
```json
{
  "question": "How many players can play?",
  "use_semantic": false
}
```

**Response:**
```json
{
  "answer": "The game supports 1-4 players...",
  "source": "rulebook",
  "page": 5,
  "confidence": 0.85,
  "scores": {
    "overall": 0.82,
    "relevance": 0.89,
    "completeness": 0.85,
    "accuracy": 0.78,
    "conciseness": 0.90
  },
  "chunks_retrieved": 25,
  "chunk_details": [...],
  "processing_time": 1.23,
  "semantic_analysis_used": false
}
```

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "game": "S.T.A.L.K.E.R.: The Board Game",
  "retriever_initialized": true,
  "scorer_initialized": true
}
```

## Configuration

### Game Information
Edit `GAME_INFO` in `app.py`:
```python
GAME_INFO = {
    "name": "Your Game Name",
    "description": "Game description",
    "sources": ["Rulebook", "FAQ", "Forums"],
    "index": "your_index_name"
}
```

### Port and Host
Change in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Retriever Settings
Modify initialization in `app.py`:
```python
retriever = RulebookRetriever(
    use_reranker=True,
    use_semantic_analysis=False  # Toggle default semantic analysis
)
```

## Development

### Debug Mode
Flask debug mode is enabled by default for development. Disable for production:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

### Testing
```bash
# Health check
curl http://localhost:5000/health

# Ask a question
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many players?", "use_semantic": false}'
```

## Troubleshooting

**Issue**: "Connection refused" error
- **Solution**: Ensure Elasticsearch is running on localhost:9200

**Issue**: "Module not found" errors
- **Solution**: Install all requirements: `pip install -r requirements.txt`

**Issue**: "Index not found" error
- **Solution**: Index your rulebook first using `src/search/indexer.py`

**Issue**: Slow response times
- **Solution**: Ensure Elasticsearch is properly indexed and reranker model is loaded

## Production Deployment

For production deployment, consider:
- Use **Gunicorn** or **uWSGI** instead of Flask's development server
- Add **nginx** as reverse proxy
- Enable HTTPS
- Add authentication/authorization
- Set up proper logging
- Use environment variables for configuration
- Disable Flask debug mode

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Future Enhancements

- [ ] Support for multiple games/rulesets
- [ ] User authentication and session history
- [ ] FAQ and forums integration
- [ ] Answer feedback mechanism
- [ ] Export answers to PDF
- [ ] Mobile-responsive improvements
- [ ] Dark mode toggle
- [ ] Multi-language support
