// SafEye Frontend Application

const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const selectedFile = document.getElementById('selectedFile');
const textInput = document.getElementById('textInput');
const analyzeTextBtn = document.getElementById('analyzeTextBtn');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const resultCard = document.getElementById('resultCard');
const findingsList = document.getElementById('findingsList');
const detailsGrid = document.getElementById('detailsGrid');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleFileDrop);

    // Text input
    textInput.addEventListener('input', () => {
        analyzeTextBtn.disabled = !textInput.value.trim();
    });

    // Analyze text button
    analyzeTextBtn.addEventListener('click', analyzeText);
}

function switchTab(tabName) {
    // Remove active class from all tabs
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });

    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    // Add active class to selected tab
    document.querySelector(`button[onclick="switchTab('${tabName}')"]`).classList.add('active');

    // Show selected tab content
    document.getElementById(`${tabName}-tab`).classList.add('active');

    // Hide results
    hideResults();
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        displaySelectedFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.style.borderColor = '#667eea';
    uploadArea.style.backgroundColor = '#f0f4ff';
}

function handleFileDrop(event) {
    event.preventDefault();
    uploadArea.style.borderColor = '#d1d5db';
    uploadArea.style.backgroundColor = '#fafafa';

    const file = event.dataTransfer.files[0];
    if (file) {
        fileInput.files = event.dataTransfer.files;
        displaySelectedFile(file);
    }
}

function displaySelectedFile(file) {
    selectedFile.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span>📄</span>
            <span style="font-weight: 500; color: #374151;">Selected: ${file.name}</span>
        </div>
    `;
    selectedFile.style.display = 'block';

    // Auto-analyze based on file type
    analyzeFile(file);
}

function hideResults() {
    progressSection.style.display = 'none';
    resultsSection.style.display = 'none';
    selectedFile.style.display = 'none';
}

async function analyzeFile(file) {
    // Show progress
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', file);

        let endpoint;
        if (file.type.startsWith('image/')) {
            endpoint = '/analyze/image';
        } else if (file.type.startsWith('audio/')) {
            endpoint = '/analyze/audio';
        } else {
            throw new Error('Unsupported file type');
        }

        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Analysis error:', error);
        alert('Error analyzing file. Please try again.');
    }

    // Hide progress
    progressSection.style.display = 'none';
}

async function analyzeText() {
    const text = textInput.value.trim();
    if (!text) return;

    // Disable button
    analyzeTextBtn.disabled = true;
    analyzeTextBtn.textContent = 'Analyzing...';

    // Show progress
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE_URL}/analyze/text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Analysis error:', error);
        alert('Error analyzing text. Please try again.');
    }

    // Hide progress and re-enable button
    progressSection.style.display = 'none';
    analyzeTextBtn.disabled = false;
    analyzeTextBtn.textContent = '🔍 Analyze Text';
}

function displayResults(result) {
    // Show results section
    resultsSection.style.display = 'block';

    // Determine result type
    const riskScore = result.risk_score;
    let resultClass, resultIcon, resultTitle;

    if (riskScore < 30) {
        resultClass = 'authentic';
        resultIcon = '✅';
        resultTitle = 'Likely Authentic';
    } else if (riskScore < 60) {
        resultClass = 'medium';
        resultIcon = '⚠️';
        resultTitle = 'Suspicious Content';
    } else {
        resultClass = 'suspicious';
        resultIcon = '❌';
        resultTitle = 'Manipulation Detected';
    }

    // Update result card
    resultCard.className = `result-card ${resultClass}`;

    // Update result header
    document.getElementById('resultTitle').textContent = resultTitle;
    document.getElementById('resultConfidence').textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
    document.getElementById('resultIcon').textContent = resultIcon;

    // Update risk meter
    document.getElementById('riskValue').textContent = `${riskScore.toFixed(1)}/100`;

    const riskFill = document.getElementById('riskFill');
    riskFill.style.width = `${riskScore}%`;

    if (riskScore < 30) {
        riskFill.className = 'risk-fill low';
    } else if (riskScore < 60) {
        riskFill.className = 'risk-fill medium';
    } else {
        riskFill.className = 'risk-fill high';
    }

    // Update findings
    findingsList.innerHTML = '';
    result.findings.forEach(finding => {
        const li = document.createElement('li');
        li.textContent = finding;
        findingsList.appendChild(li);
    });

    // Update findings card class for styling
    const findingsCard = document.querySelector('.findings-card');
    findingsCard.classList.remove('suspicious');
    if (resultClass === 'suspicious') {
        findingsCard.classList.add('suspicious');
    }

    // Update details
    detailsGrid.innerHTML = '';
    if (result.details) {
        Object.entries(result.details).forEach(([key, value]) => {
            const detailItem = document.createElement('div');
            detailItem.className = 'detail-item';
            detailItem.innerHTML = `
                <span class="detail-label">${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</span>
                <span class="detail-value">${value}</span>
            `;
            detailsGrid.appendChild(detailItem);
        });
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getFileIcon(type) {
    if (type.startsWith('image/')) return '🖼️';
    if (type.startsWith('audio/')) return '🎵';
    if (type.startsWith('video/')) return '🎥';
    return '📄';
}

// Error handling
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    alert('An unexpected error occurred. Please refresh the page and try again.');
});

window.addEventListener('error', function(event) {
    console.error('JavaScript error:', event.error);
    alert('An unexpected error occurred. Please refresh the page and try again.');
});
