// Main application logic
const questionForm = document.getElementById('question-form');
const questionInput = document.getElementById('question-input');
const semanticToggle = document.getElementById('semantic-toggle');
const askButton = document.getElementById('ask-button');
const loadingSection = document.getElementById('loading');
const answerSection = document.getElementById('answer-section');
const errorSection = document.getElementById('error-section');

// Answer elements
const answerContent = document.getElementById('answer-content');
const sourceBadge = document.getElementById('source-badge');
const pageInfo = document.getElementById('page-info');
const confidenceBadge = document.getElementById('confidence-badge');
const scoreGrid = document.getElementById('score-grid');

// Debug elements
const debugTime = document.getElementById('debug-time');
const debugChunksCount = document.getElementById('debug-chunks-count');
const debugSemantic = document.getElementById('debug-semantic');
const debugChunks = document.getElementById('debug-chunks');
const errorMessage = document.getElementById('error-message');

// Form submission handler
questionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const question = questionInput.value.trim();
    if (!question) {
        return;
    }
    
    // Show loading, hide previous results
    loadingSection.style.display = 'block';
    answerSection.style.display = 'none';
    errorSection.style.display = 'none';
    askButton.disabled = true;
    
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                use_semantic: semanticToggle.checked
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Unknown error occurred');
        }
        
        const data = await response.json();
        displayAnswer(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    } finally {
        loadingSection.style.display = 'none';
        askButton.disabled = false;
    }
});

// Display answer and metadata
function displayAnswer(data) {
    // Display spellcheck corrections if any
    if (data.spellcheck_corrections && data.spellcheck_corrections.length > 0) {
        displaySpellcheckCorrections(data.spellcheck_corrections, data.original_question, data.corrected_question);
    } else {
        // Hide corrections section if no corrections
        const correctionsSection = document.getElementById('spellcheck-corrections');
        if (correctionsSection) {
            correctionsSection.style.display = 'none';
        }
    }
    
    // Answer text
    answerContent.textContent = data.answer;
    
    // Source and page info
    sourceBadge.textContent = capitalizeFirst(data.source);
    pageInfo.textContent = data.page ? `Page: ${data.page}` : 'Page: N/A';
    
    // Confidence badge
    const confidence = data.confidence * 100;
    confidenceBadge.textContent = `Confidence: ${confidence.toFixed(0)}%`;
    confidenceBadge.className = 'confidence-badge';
    if (confidence >= 70) {
        confidenceBadge.classList.add('high');
    } else if (confidence >= 50) {
        confidenceBadge.classList.add('medium');
    } else {
        confidenceBadge.classList.add('low');
    }
    
    // Quality scores
    displayScores(data.scores);
    
    // Debug information
    debugTime.textContent = data.processing_time;
    debugChunksCount.textContent = data.chunks_retrieved;
    debugSemantic.textContent = data.semantic_analysis_used ? 'Enabled' : 'Disabled';
    
    // Display semantic debug info if available
    if (data.semantic_debug) {
        displaySemanticDebug(data.semantic_debug);
    } else {
        document.getElementById('semantic-debug-section').style.display = 'none';
    }
    
    // Chunk details
    displayChunks(data.chunk_details);
    
    // Show answer section
    answerSection.style.display = 'block';
    
    // Scroll to answer
    answerSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display quality scores
function displayScores(scores) {
    scoreGrid.innerHTML = '';
    
    const scoreLabels = {
        overall: 'Overall',
        relevance: 'Relevance',
        completeness: 'Completeness',
        accuracy: 'Accuracy',
        conciseness: 'Conciseness'
    };
    
    for (const [key, label] of Object.entries(scoreLabels)) {
        const scoreValue = scores[key] * 100;
        const scoreItem = document.createElement('div');
        scoreItem.className = 'score-item';
        
        const scoreLabel = document.createElement('div');
        scoreLabel.className = 'score-label';
        scoreLabel.textContent = label;
        
        const scoreValueEl = document.createElement('div');
        scoreValueEl.className = 'score-value';
        scoreValueEl.textContent = `${scoreValue.toFixed(0)}%`;
        
        // Color coding
        if (scoreValue >= 80) {
            scoreValueEl.classList.add('high');
        } else if (scoreValue >= 60) {
            scoreValueEl.classList.add('medium');
        } else {
            scoreValueEl.classList.add('low');
        }
        
        scoreItem.appendChild(scoreLabel);
        scoreItem.appendChild(scoreValueEl);
        scoreGrid.appendChild(scoreItem);
    }
}

// Display chunk details
function displayChunks(chunks) {
    debugChunks.innerHTML = '';
    
    if (!chunks || chunks.length === 0) {
        debugChunks.innerHTML = '<p>No chunk details available</p>';
        return;
    }
    
    chunks.forEach((chunk, index) => {
        const chunkItem = document.createElement('div');
        chunkItem.className = 'chunk-item';
        
        // Header with rank and score
        const header = document.createElement('div');
        header.className = 'chunk-header';
        
        const rank = document.createElement('span');
        rank.className = 'chunk-rank';
        rank.textContent = `#${chunk.rank}`;
        
        const score = document.createElement('span');
        score.className = 'chunk-score';
        score.textContent = `Score: ${chunk.score.toFixed(4)}`;
        
        header.appendChild(rank);
        header.appendChild(score);
        
        // Metadata
        const meta = document.createElement('div');
        meta.className = 'chunk-meta';
        meta.textContent = `Page: ${chunk.page || 'N/A'} | Section: ${chunk.section}`;
        
        // Text content
        const text = document.createElement('div');
        text.className = 'chunk-text';
        text.textContent = chunk.text;
        
        chunkItem.appendChild(header);
        chunkItem.appendChild(meta);
        chunkItem.appendChild(text);
        
        debugChunks.appendChild(chunkItem);
    });
}

// Display semantic debug information
function displaySemanticDebug(semanticDebug) {
    const section = document.getElementById('semantic-debug-section');
    
    if (semanticDebug.error) {
        section.innerHTML = `<p class="error">Semantic analysis error: ${semanticDebug.error}</p>`;
        section.style.display = 'block';
        return;
    }
    
    // Show section
    section.style.display = 'block';
    
    // Update all semantic debug fields
    document.getElementById('semantic-original').textContent = semanticDebug.original_query || '-';
    document.getElementById('semantic-enhanced').textContent = semanticDebug.enhanced_query || '-';
    document.getElementById('semantic-question-type').textContent = semanticDebug.question_type || '-';
    
    // Format arrays as comma-separated lists
    document.getElementById('semantic-query-words').textContent = 
        (semanticDebug.query_words && semanticDebug.query_words.length > 0) 
            ? semanticDebug.query_words.join(', ') 
            : '-';
    
    document.getElementById('semantic-key-nouns').textContent = 
        (semanticDebug.key_nouns && semanticDebug.key_nouns.length > 0) 
            ? semanticDebug.key_nouns.join(', ') 
            : '-';
    
    document.getElementById('semantic-action-verbs').textContent = 
        (semanticDebug.action_verbs && semanticDebug.action_verbs.length > 0) 
            ? semanticDebug.action_verbs.join(', ') 
            : '-';
    
    document.getElementById('semantic-game-concepts').textContent = 
        (semanticDebug.game_concepts && semanticDebug.game_concepts.length > 0) 
            ? semanticDebug.game_concepts.join(', ') 
            : '-';
    
    document.getElementById('semantic-domain-keywords').textContent = 
        (semanticDebug.domain_keywords && semanticDebug.domain_keywords.length > 0) 
            ? semanticDebug.domain_keywords.join(', ') 
            : '-';
    
    // Format intent flags
    const intentFlags = semanticDebug.intent_flags || {};
    const activeFlags = Object.entries(intentFlags)
        .filter(([key, value]) => value)
        .map(([key, value]) => key);
    
    document.getElementById('semantic-intent-flags').textContent = 
        activeFlags.length > 0 ? activeFlags.join(', ') : 'none';
}

// Toggle semantic details visibility
function toggleSemanticDetails() {
    const content = document.getElementById('semantic-details-content');
    const toggle = document.getElementById('semantic-toggle');
    if (content.style.display === 'none') {
        content.style.display = 'block';
        toggle.textContent = '▼';
    } else {
        content.style.display = 'none';
        toggle.textContent = '▶';
    }
}

// Display spellcheck corrections
function displaySpellcheckCorrections(corrections, originalQuestion, correctedQuestion) {
    // Ensure corrections is an array
    if (!Array.isArray(corrections) || corrections.length === 0) {
        return;
    }
    
    let correctionsSection = document.getElementById('spellcheck-corrections');
    if (!correctionsSection) {
        // Create corrections section if it doesn't exist
        const section = document.createElement('div');
        section.id = 'spellcheck-corrections';
        section.className = 'spellcheck-corrections';
        
        // Insert before answer section
        const answerSection = document.getElementById('answer-section');
        answerSection.parentNode.insertBefore(section, answerSection);
        correctionsSection = section;
    }
    
    correctionsSection.style.display = 'block';
    
    // Build corrections message - handle both tuple arrays and objects
    const correctionsList = corrections.map(item => {
        if (Array.isArray(item) && item.length >= 2) {
            return `"${item[0]}" → "${item[1]}"`;
        } else if (typeof item === 'object' && item.original && item.corrected) {
            return `"${item.original}" → "${item.corrected}"`;
        }
        return '';
    }).filter(s => s).join(', ');
    
    correctionsSection.innerHTML = `
        <div class="spellcheck-icon">✓</div>
        <div class="spellcheck-content">
            <div class="spellcheck-title">Spelling corrections applied</div>
            <div class="spellcheck-details">
                ${correctionsList}
            </div>
            <div class="spellcheck-queries">
                <div><strong>Original:</strong> ${originalQuestion || 'N/A'}</div>
                <div><strong>Corrected:</strong> ${correctedQuestion || 'N/A'}</div>
            </div>
        </div>
    `;
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
}

// Helper function
function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Example questions
const exampleQuestions = [
    "How many players can play?",
    "What happens when a stalker dies?",
    "How does combat work?",
    "What are anomalies?",
    "How do I win the game?"
];

// Add keyboard shortcut (Ctrl+Enter to submit)
questionInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        questionForm.dispatchEvent(new Event('submit'));
    }
});

console.log('Boardgame Rules Checker UI initialized');
console.log('Try asking:', exampleQuestions[Math.floor(Math.random() * exampleQuestions.length)]);
