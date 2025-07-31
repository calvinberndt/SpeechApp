// app.js - Voice AI Assistant Script

let isRecording = false;
let websocket;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 3000;

// Media Recording Variables
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let audioStream = null;
let recordingTimeout = null;
const MAX_RECORDING_TIME = 300000; // 5 minutes

// Feature Detection for Browser Compatibility
const isMediaRecorderSupported = 'MediaRecorder' in window;
const isWebAudioSupported = 'AudioContext' in window || 'webkitAudioContext' in window;
const isGetUserMediaSupported = navigator.mediaDevices && 'getUserMedia' in navigator.mediaDevices;
const isWebRTCSupported = !!window.RTCPeerConnection;
const isWebSocketSupported = 'WebSocket' in window;

// Browser Compatibility Check
function checkBrowserCompatibility() {
    const incompatibilities = [];
    
    if (!isWebSocketSupported) {
        incompatibilities.push('WebSocket');
    }
    
    if (!isGetUserMediaSupported) {
        incompatibilities.push('MediaDevices getUserMedia');
    }
    
    if (!isMediaRecorderSupported) {
        incompatibilities.push('MediaRecorder');
    }
    
    if (!isWebRTCSupported) {
        incompatibilities.push('WebRTC');
    }
    
    if (!isWebAudioSupported) {
        incompatibilities.push('Web Audio API');
    }
    
    return {
        isSupported: incompatibilities.length === 0,
        missing: incompatibilities
    };
}

// UI Elements
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const aiOrb = document.getElementById('ai-orb');
const messagesContainer = document.getElementById('messages');
const textInput = document.getElementById('text-input');
const audioPlayback = document.getElementById('audio-playback');

// Initialize audio recording
function initAudioRecording() {
    if (!isGetUserMediaSupported) {
        updateStatus('error', 'getUserMedia is not supported in your browser');
        return;
    }

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            audioStream = stream;
            mediaRecorder = new MediaRecorder(stream);
            updateStatus('ready', 'Microphone access granted. Ready to record.');
        })
        .catch(error => {
            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                updateStatus('error', 'Microphone access denied. Please allow access and try again.');
            } else {
                updateStatus('error', 'An error occurred while accessing the microphone.');
                console.error('getUserMedia error:', error);
            }
        });
}

// Initialize WebSocket connection
function initWebSocket() {
    const clientId = `user-${Math.floor(Math.random() * 10000)}`;
    websocket = new WebSocket(`ws://${window.location.host}/ws/${clientId}`);

    websocket.onopen = function() {
        console.log('WebSocket connection established');
    };

    websocket.onmessage = function(event) {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };

    websocket.onclose = function() {
        console.log('WebSocket connection closed');
    };

    websocket.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
}

// Handle incoming WebSocket messages
function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'status_update':
            updateStatus(message.status, message.message);
            break;
        case 'user_speech':
            displayMessage('Calvin', message.text, 'user');
            break;
        case 'ai_response':
            displayMessage('AI', message.text, 'ai');
            break;
        case 'ai_audio':
            playAudio(message.audio_url);
            break;
        case 'error':
            alert(message.message);
            break;
        default:
            console.warn('Unknown message type:', message.type);
            break;
    }
}

// Update the status indicator and text
function updateStatus(status, message) {
    statusIndicator.className = `status-indicator ${status}`;
    statusText.textContent = message;
    aiOrb.className = `ai-orb ${status}`;
}

// Display a message in the conversation
function displayMessage(sender, text, type) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${type}`;
    const header = document.createElement('div');
    header.className = 'message-header';
    header.textContent = sender;
    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = text;
    messageElement.appendChild(header);
    messageElement.appendChild(content);
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Play audio from URL
function playAudio(url) {
    audioPlayback.src = url;
audioPlayback.play().catch(error => {
        updateStatus('error', 'Audio playback error: ' + error.name);
        console.error('Audio playback error:', error);
    });
}

// Initialize event listeners
function initEventListeners() {
    const voiceBtn = document.getElementById('voice-btn');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');
    const settingsToggleBtn = document.getElementById('settings-toggle-btn');
    const settingsPanel = document.getElementById('settings-panel');

    // Toggle recording state when the voice button is pressed
    voiceBtn.addEventListener('mousedown', startRecording);
    voiceBtn.addEventListener('mouseup', stopRecording);
    voiceBtn.addEventListener('mouseleave', stopRecording);

    // Send text input when the send button is pressed
    sendBtn.addEventListener('click', () => {
        const message = textInput.value.trim();
        if (message) {
            websocket.send(JSON.stringify({ type: 'text_input', text: message }));
            textInput.value = '';
        }
    });
    
    // Send text input when Enter is pressed
    textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const message = textInput.value.trim();
            if (message) {
                websocket.send(JSON.stringify({ type: 'text_input', text: message }));
                textInput.value = '';
            }
        }
    });

    // Clear conversation when the clear button is pressed
    clearBtn.addEventListener('click', () => {
        messagesContainer.innerHTML = '';
    });

    // Toggle settings panel
    settingsToggleBtn.addEventListener('click', () => {
        settingsPanel.classList.toggle('show');
    });
}

// Convert audio Blob to a base64 string
function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result.split(',')[1]; // Remove data URL prefix
            resolve(base64String);
        };
        reader.onerror = (error) => {
            console.error('Error converting Blob to Base64:', error);
            reject('Failed to convert Blob to Base64.');
        };
        reader.readAsDataURL(blob);
    });
}

// Start audio recording
function startRecording() {
    if (!isMediaRecorderSupported) {
        updateStatus('error', 'MediaRecorder is not supported in your browser');
        return;
    }

    if (!audioStream) {
        updateStatus('error', 'Audio stream is not initialized');
        return;
    }

    // Clear previous audio chunks
    audioChunks = [];

    // Create a MediaRecorder instance
try {
        mediaRecorder = new MediaRecorder(audioStream);
    } catch (error) {
        updateStatus('error', 'Failed to initialize MediaRecorder.');
        console.error('MediaRecorder initialization error:', error);
        return;
    }

    // Set up event handlers
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
        }
    };

mediaRecorder.onerror = (event) => {
        updateStatus('error', 'Recording error: ' + event.error);
        console.error('Recording error:', event.error);
    };

    mediaRecorder.onstop = () => {
        // This will be handled in stopRecording() function
    };

    // Start recording
    mediaRecorder.start();
    isRecording = true;
    updateStatus('listening', 'Listening...');
    console.log('Recording started');
}

// Stop audio recording and send data
function stopRecording() {
    if (isRecording && mediaRecorder && mediaRecorder.state === 'recording') {
        isRecording = false;
        updateStatus('processing', 'Processing speech...');
        console.log('Recording stopped');
        
        // Stop the MediaRecorder
        mediaRecorder.stop();
        
        // Process the audio data
        setTimeout(() => {
            if (audioChunks.length > 0) {
                // Convert recorded audio chunks to a Blob
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                
                // Convert the Blob to base64 format
                const reader = new FileReader();
                reader.onloadend = function() {
                    const base64Audio = reader.result.split(',')[1]; // Remove data URL prefix
                    
                    // Send the base64 audio data to the WebSocket server
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(JSON.stringify({
                            type: 'audio_data',
                            audio: base64Audio,
                            format: 'webm'
                        }));
                        console.log('Audio data sent to server');
                    } else {
                        updateStatus('error', 'WebSocket connection is not available');
                        console.error('WebSocket is not connected');
                    }
                };
                reader.readAsDataURL(audioBlob);
            } else {
                updateStatus('error', 'No audio data recorded');
                console.warn('No audio chunks available');
            }
        }, 100); // Small delay to ensure MediaRecorder has finished
    }
}

// Initialize application
window.onload = function() {
    initWebSocket();
    initEventListeners();
const compatibility = checkBrowserCompatibility();
    if (!compatibility.isSupported) {
        updateStatus('error', 'Incompatible browser. Missing: ' + compatibility.missing.join(', '));
        return;
    }
    initAudioRecording();
};

