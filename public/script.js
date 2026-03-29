const API_URL = 'http://localhost:8000';
const chatContainer = document.getElementById('chat-container');
const messageTemplate = document.getElementById('message-template');
const form = document.getElementById('chat-form');
const inputField = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');

let isWaitingForResponse = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    // Recheck health periodically
    setInterval(checkHealth, 30000);
});

// Set input value from quick suggestions
function setInput(text) {
    inputField.value = text;
    form.dispatchEvent(new Event('submit'));
}

async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/health`);
        if (res.ok) {
            connectionDot.className = 'dot connected';
            connectionText.textContent = 'API Connected';
        } else {
            throw new Error('API not healthy');
        }
    } catch (err) {
        connectionDot.className = 'dot error';
        connectionText.textContent = 'API Disconnected';
        console.error('Health check failed', err);
    }
}

async function handleQuery(e) {
    e.preventDefault();
    if (isWaitingForResponse) return;

    const query = inputField.value.trim();
    if (!query) return;

    // Remove welcome message if it exists
    const welcomeArea = document.querySelector('.welcome-message');
    if (welcomeArea) {
        welcomeArea.style.display = 'none';
    }

    addMessage(query, 'user');
    inputField.value = '';
    
    // Disable inputs and show loading
    isWaitingForResponse = true;
    inputField.disabled = true;
    sendButton.disabled = true;
    
    const loadingId = addLoadingIndicator();

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        removeLoadingIndicator(loadingId);
        
        const answerHtml = window.marked && window.marked.parse ? window.marked.parse(data.answer) : data.answer.replace(/\n/g, '<br>');
        
        addMessage(answerHtml, 'assistant', data.sources);

    } catch (error) {
        console.error('Query error:', error);
        removeLoadingIndicator(loadingId);
        addMessage(`**Error:** Unable to get an answer. Please check if the API is running correctly. (${error.message})`, 'assistant');
    } finally {
        isWaitingForResponse = false;
        inputField.disabled = false;
        sendButton.disabled = false;
        inputField.focus();
    }
}

function addMessage(content, role, sources = []) {
    const clone = messageTemplate.content.cloneNode(true);
    const messageEl = clone.querySelector('.message');
    const contentEl = clone.querySelector('.message-text');
    
    messageEl.classList.add(role);
    
    // Parse markdown if it's the assistant, otherwise escape HTML (user)
    if (role === 'user') {
        contentEl.textContent = content; // textContent escapes HTML
    } else {
        contentEl.innerHTML = window.marked && window.marked.parse ? window.marked.parse(content) : content;
    }

    // Handle sources
    if (sources && sources.length > 0) {
        const sourcesContainer = clone.querySelector('.message-sources');
        const sourcesList = clone.querySelector('.sources-list');
        const toggleBtn = clone.querySelector('.sources-toggle');
        
        sourcesContainer.style.display = 'block';
        
        sources.forEach((src, idx) => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${src.source}</strong> (Page ${src.page || 'N/A'})<br><em>"${src.snippet.substring(0, 100)}..."</em>`;
            sourcesList.appendChild(li);
        });

        toggleBtn.addEventListener('click', () => {
            sourcesContainer.classList.toggle('open');
        });
    }

    chatContainer.appendChild(messageEl);
    scrollToBottom();
}

function addLoadingIndicator() {
    const id = 'loading-' + Date.now();
    const clone = messageTemplate.content.cloneNode(true);
    const messageEl = clone.querySelector('.message');
    const contentEl = clone.querySelector('.message-content');
    
    messageEl.classList.add('assistant');
    messageEl.id = id;
    
    // Replace text area with loading bubbles
    contentEl.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    
    chatContainer.appendChild(messageEl);
    scrollToBottom();
    return id;
}

function removeLoadingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    chatContainer.scrollTo({
        top: chatContainer.scrollHeight,
        behavior: 'smooth'
    });
}
