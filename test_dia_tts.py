#!/usr/bin/env python3
"""
Test script for Dia TTS integration
"""

import asyncio
import logging
from backend.tts_processor import TTSProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_dia_tts():
    """Test the Dia TTS processor"""
    
    print("Testing Dia TTS integration...")
    
    # Initialize TTS processor
    tts_processor = TTSProcessor()
    
    # Initialize the model
    print("Initializing Dia TTS model...")
    await tts_processor._initialize()
    
    if not tts_processor.is_ready():
        print("❌ TTS processor failed to initialize")
        return
        
    print("✅ TTS processor initialized successfully")
    
    # Test text-to-speech conversion
    test_text = "Hello! This is a test of the Dia TTS model integration. The model should generate natural-sounding speech from this text."
    
    print(f"Converting text to speech: '{test_text[:50]}...'")
    
    try:
        audio_file = await tts_processor.text_to_speech(test_text)
        
        if audio_file and audio_file.exists():
            print(f"✅ Audio file generated successfully: {audio_file}")
            print(f"File size: {audio_file.stat().st_size} bytes")
        else:
            print("❌ Audio file generation failed")
            
    except Exception as e:
        print(f"❌ Error during TTS conversion: {e}")

if __name__ == "__main__":
    asyncio.run(test_dia_tts())
