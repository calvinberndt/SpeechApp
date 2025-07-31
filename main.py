"""
Voice AI Assistant - Main Application
Combines Kyutai STT, Local LLM (Ollama), and Kyutai TTS
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from backend.speech_processor import SpeechProcessor
from backend.llm_client import LLMClient
from backend.tts_processor import TTSProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Voice AI Assistant", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize processors
speech_processor = SpeechProcessor()
llm_client = LLMClient()
tts_processor = TTSProcessor()

# Initialize processors async function
async def initialize_processors():
    """Initialize all processors asynchronously"""
    await speech_processor._initialize()
    await llm_client._initialize()
    await tts_processor._initialize()

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

@app.get("/", response_class=HTMLResponse)
async def get_main_page(request: Request):
    """Serve the main application page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle WebSocket connections for real-time communication"""
    await websocket.accept()
    active_connections[client_id] = websocket
    logger.info(f"Client {client_id} connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_websocket_message(websocket, client_id, message)
            
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        active_connections.pop(client_id, None)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        active_connections.pop(client_id, None)

async def handle_websocket_message(websocket: WebSocket, client_id: str, message: dict):
    """Process incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "audio_data":
        # Process audio data for STT
        await process_audio_message(websocket, client_id, message)
    elif message_type == "text_input":
        # Process direct text input
        await process_text_message(websocket, client_id, message)
    else:
        logger.warning(f"Unknown message type: {message_type}")

async def process_audio_message(websocket: WebSocket, client_id: str, message: dict):
    """Process audio data through STT -> LLM -> TTS pipeline"""
    try:
        # Update UI state
        await websocket.send_text(json.dumps({
            "type": "status_update",
            "status": "processing_speech",
            "message": "Converting speech to text..."
        }))
        
        # Get audio data (base64 encoded)
        audio_data = message.get("audio_data")
        if not audio_data:
            return
        
        # Convert speech to text
        transcribed_text = await speech_processor.process_audio(audio_data)
        
        if not transcribed_text.strip():
            await websocket.send_text(json.dumps({
                "type": "status_update",
                "status": "idle",
                "message": "No speech detected"
            }))
            return
        
        # Display user's speech
        await websocket.send_text(json.dumps({
            "type": "user_speech",
            "text": transcribed_text
        }))
        
        # Update UI state
        await websocket.send_text(json.dumps({
            "type": "status_update",
            "status": "thinking",
            "message": "AI is thinking..."
        }))
        
        # Get LLM response
        ai_response = await llm_client.get_response(transcribed_text)
        
        # Display AI's response text
        await websocket.send_text(json.dumps({
            "type": "ai_response",
            "text": ai_response
        }))
        
        # Update UI state
        await websocket.send_text(json.dumps({
            "type": "status_update",
            "status": "speaking",
            "message": "AI is speaking..."
        }))
        
        # Convert AI response to speech
        audio_file_path = await tts_processor.text_to_speech(ai_response)
        
        if audio_file_path:
            # Send audio file to client
            await websocket.send_text(json.dumps({
                "type": "ai_audio",
                "audio_url": f"/audio/{Path(audio_file_path).name}"
            }))
        else:
            logger.warning("TTS failed, no audio generated")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Text-to-speech conversion failed"
            }))
        
        # Reset to idle state
        await websocket.send_text(json.dumps({
            "type": "status_update",
            "status": "idle",
            "message": "Ready to listen..."
        }))
        
    except Exception as e:
        logger.error(f"Error processing audio message: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Error processing audio: {str(e)}"
        }))

async def process_text_message(websocket: WebSocket, client_id: str, message: dict):
    """Process direct text input through LLM -> TTS pipeline"""
    try:
        text_input = message.get("text", "").strip()
        if not text_input:
            return
        
        # Display user's text
        await websocket.send_text(json.dumps({
            "type": "user_speech",
            "text": text_input
        }))
        
        # Update UI state
        await websocket.send_text(json.dumps({
            "type": "status_update",
            "status": "thinking",
            "message": "AI is thinking..."
        }))
        
        # Get LLM response
        ai_response = await llm_client.get_response(text_input)
        
        # Display AI's response
        await websocket.send_text(json.dumps({
            "type": "ai_response",
            "text": ai_response
        }))
        
        # Convert to speech
        await websocket.send_text(json.dumps({
            "type": "status_update",
            "status": "speaking",
            "message": "AI is speaking..."
        }))
        
        audio_file_path = await tts_processor.text_to_speech(ai_response)
        
        if audio_file_path:
            await websocket.send_text(json.dumps({
                "type": "ai_audio",
                "audio_url": f"/audio/{Path(audio_file_path).name}"
            }))
        else:
            logger.warning("TTS failed, no audio generated")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Text-to-speech conversion failed"
            }))
        
        await websocket.send_text(json.dumps({
            "type": "status_update",
            "status": "idle",
            "message": "Ready to listen..."
        }))
        
    except Exception as e:
        logger.error(f"Error processing text message: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Error processing text: {str(e)}"
        }))

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files"""
    file_path = Path("audio_files") / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="audio/wav")
    return {"error": "File not found"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "stt": speech_processor.is_ready(),
            "llm": await llm_client.is_ready(),
            "tts": tts_processor.is_ready()
        }
    }

async def startup_event():
    """Initialize processors on startup"""
    await initialize_processors()

# Add startup event
app.add_event_handler("startup", startup_event)

if __name__ == "__main__":
    # Ensure audio files directory exists
    Path("audio_files").mkdir(exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
