// Configuration
const API_BASE = window.location.origin;
let currentResult = null;

// DOM Elements
const expenseInput = document.getElementById('expenseInput');
const categorizeBtn = document.getElementById('categorizeBtn');
const clearBtn = document.getElementById('clearBtn');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultDisplay = document.getElementById('resultDisplay');
const copyBtn = document.getElementById('copyBtn');
const newBtn = document.getElementById('newBtn');
const saveBtn = document.getElementById('saveBtn');
const modelStatus = document.getElementById('modelStatus');
const exampleChips = document.querySelectorAll('.example-chip');

// Category Configuration
const CATEGORY_CONFIG = {
    food: {
        name: 'Food & Dining',
        icon: 'fa-utensils',
        color: '#4CAF50',
        gradient: 'linear-gradient(135deg, #4CAF50, #2E7D32)'
    },
    transport: {
        name: 'Transportation',
        icon: 'fa-car',
        color: '#2196F3',
        gradient: 'linear-gradient(135deg, #2196F3, #0D47A1)'
    },
    shopping: {
        name: 'Shopping',
        icon: 'fa-shopping-bag',
        color: '#FF9800',
        gradient: 'linear-gradient(135deg, #FF9800, #EF6C00)'
    },
    utilities: {
        name: 'Utilities',
        icon: 'fa-bolt',
        color: '#9C27B0',
        gradient: 'linear-gradient(135deg, #9C27B0, #6A1B9A)'
    },
    entertainment: {
        name: 'Entertainment',
        icon: 'fa-film',
        color: '#E91E63',
        gradient: 'linear-gradient(135deg, #E91E63, #AD1457)'
    },
    other: {
        name: 'Other',
        icon: 'fa-question-circle',
        color: '#757575',
        gradient: 'linear-gradient(135deg, #757575, #424242)'
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Check API health
    checkAPIHealth();
    
    // Setup event listeners
    setupEventListeners();
    
    // Focus on input
    expenseInput.focus();
});

// Setup Event Listeners
function setupEventListeners() {
    // Categorize button
    categorizeBtn.addEventListener('click', handleCategorize);
    
    // Enter key in input
    expenseInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleCategorize();
        }
    });
    
    // Clear input button
    clearBtn.addEventListener('click', function() {
        expenseInput.value = '';
        expenseInput.focus();
        showToast('Input cleared', 'info');
    });
    
    // Example chips
    exampleChips.forEach(chip => {
        chip.addEventListener('click', function() {
            const text = this.getAttribute('data-text');
            expenseInput.value = text;
            handleCategorize();
        });
    });
    
    // Action buttons
    copyBtn.addEventListener('click', copyResult);
    newBtn.addEventListener('click', resetForm);
    saveBtn.addEventListener('click', saveResult);
    
    // Input validation
    expenseInput.addEventListener('input', function() {
        categorizeBtn.disabled = this.value.trim().length === 0;
    });
}

// Check API Health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        
        if (data.ml_model_loaded) {
            modelStatus.innerHTML = '<i class="fas fa-circle" style="color: #4CAF50"></i> ML Model Ready';
            modelStatus.style.color = '#4CAF50';
        } else {
            modelStatus.innerHTML = '<i class="fas fa-circle" style="color: #FF9800"></i> Using Keyword Fallback';
            modelStatus.style.color = '#FF9800';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        modelStatus.innerHTML = '<i class="fas fa-circle" style="color: #f44336"></i> API Offline';
        modelStatus.style.color = '#f44336';
    }
}

// Handle Categorization
async function handleCategorize() {
    const text = expenseInput.value.trim();
    
    if (!text) {
        showToast('Please enter an expense description', 'warning');
        return;
    }
    
    // Show loading state
    categorizeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    categorizeBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/api/categorize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentResult = data.result;
            displayResult(currentResult);
            showToast('Expense categorized successfully!', 'success');
        } else {
            showToast(data.error || 'Categorization failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Failed to connect to server', 'error');
    } finally {
        // Reset button
        categorizeBtn.innerHTML = '<i class="fas fa-brain"></i> Analyze with AI';
        categorizeBtn.disabled = false;
    }
}

// Display Result
function displayResult(result) {
    const config = CATEGORY_CONFIG[result.category] || CATEGORY_CONFIG.other;
    
    // Hide placeholder, show result
    resultPlaceholder.style.display = 'none';
    resultDisplay.style.display = 'block';
    
    // Update category badge
    const categoryBadge = document.querySelector('.category-badge');
    categoryBadge.className = 'category-badge category-' + result.category;
    categoryBadge.style.background = config.gradient;
    
    // Update icon
    document.getElementById('categoryIcon').innerHTML = `<i class="fas ${config.icon}"></i>`;
    
    // Update text
    document.getElementById('categoryName').textContent = config.name;
    document.getElementById('categoryExplanation').textContent = result.explanation;
    
    // Update confidence
    const confidencePercent = Math.round(result.confidence * 100);
    document.getElementById('confidenceValue').textContent = `${confidencePercent}%`;
    
    // Update details
    document.getElementById('inputText').textContent = result.text;
    document.getElementById('modelUsed').textContent = result.model_used;
    document.getElementById('aiInsight').textContent = result.explanation;
    
    // Add special note for "going to" patterns
    if (result.category === 'transport' && result.destination) {
        const insight = document.getElementById('aiInsight');
        insight.innerHTML = `${result.explanation}<br>
                            <small style="opacity:0.8">
                            <i class="fas fa-map-marker-alt"></i> 
                            Detected destination: ${result.destination}
                            </small>`;
    }
}

// Copy Result
function copyResult() {
    if (!currentResult) return;
    
    const textToCopy = `
Expense: ${currentResult.text}
Category: ${currentResult.category}
Confidence: ${Math.round(currentResult.confidence * 100)}%
Explanation: ${currentResult.explanation}
    `.trim();
    
    navigator.clipboard.writeText(textToCopy)
        .then(() => showToast('Result copied to clipboard!', 'success'))
        .catch(() => showToast('Failed to copy', 'error'));
}

// Reset Form
function resetForm() {
    expenseInput.value = '';
    expenseInput.focus();
    resultDisplay.style.display = 'none';
    resultPlaceholder.style.display = 'block';
    currentResult = null;
    showToast('Ready for new analysis', 'info');
}

// Save Result
function saveResult() {
    if (!currentResult) return;
    
    // In a real app, this would save to database
    const savedExpenses = JSON.parse(localStorage.getItem('expenses') || '[]');
    savedExpenses.push({
        ...currentResult,
        saved_at: new Date().toISOString(),
        id: Date.now()
    });
    
    localStorage.setItem('expenses', JSON.stringify(savedExpenses));
    showToast('Expense saved locally!', 'success');
}

// Toast Notification
function showToast(message, type = 'info') {
    // Remove existing toasts
    const existingToasts = document.querySelectorAll('.toast');
    existingToasts.forEach(toast => toast.remove());
    
    // Create toast
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas ${getToastIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add styles
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${getToastColor(type)};
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 15px;
        min-width: 300px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    // Add animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .toast-content {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .toast-close {
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            font-size: 14px;
            opacity: 0.8;
        }
        
        .toast-close:hover {
            opacity: 1;
        }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 3000);
}

function getToastIcon(type) {
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    return icons[type] || 'fa-info-circle';
}

function getToastColor(type) {
    const colors = {
        success: '#4CAF50',
        error: '#f44336',
        warning: '#FF9800',
        info: '#2196F3'
    };
    return colors[type] || '#2196F3';
}

// Category items hover effect
document.querySelectorAll('.category-item').forEach(item => {
    item.addEventListener('click', function() {
        const category = this.getAttribute('data-category');
        const examples = {
            food: 'dinner at restaurant',
            transport: 'going to ambala',
            shopping: 'buy new clothes',
            utilities: 'electricity bill',
            entertainment: 'movie tickets'
        };
        
        if (examples[category]) {
            expenseInput.value = examples[category];
            handleCategorize();
        }
    });
});