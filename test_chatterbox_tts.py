#!/usr/bin/env python3
"""
Test script for Chatterbox TTS integration
"""

import asyncio
import logging
import os
import soundfile as sf
from pathlib import Path
from backend.tts_processor import TTSProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_chatterbox_tts():
    """Test the Chatterbox TTS processor"""
    
    print("="*60)
    print("Testing Chatterbox TTS integration...")
    print("="*60)
    
    # Initialize TTS processor
    tts_processor = TTSProcessor()
    
    # Initialize the model
    print("\n1. Initializing Chatterbox TTS model...")
    await tts_processor._initialize()
    
    if not tts_processor.is_ready():
        print("❌ TTS processor failed to initialize")
        return
        
    print("✅ TTS processor initialized successfully\n")
    
    # Function to verify audio file
    def verify_audio_file(audio_file, test_name):
        if audio_file and audio_file.exists():
            file_size = audio_file.stat().st_size
            print(f"✅ {test_name}: Audio file generated successfully")
            print(f"   📁 File path: {audio_file}")
            print(f"   📊 File size: {file_size:,} bytes")
            
            # Check if it's a proper audio file
            try:
                if audio_file.suffix.lower() in ['.wav', '.aiff']:
                    if audio_file.suffix.lower() == '.wav':
                        info = sf.info(str(audio_file))
                        print(f"   🎵 Format: {info.format}, Sample rate: {info.samplerate} Hz, Duration: {info.duration:.2f}s")
                    else:
                        print(f"   🎵 Format: AIFF (macOS say fallback)")
                else:
                    print(f"   ⚠️  Unknown audio format: {audio_file.suffix}")
            except Exception as e:
                print(f"   ⚠️  Could not read audio file info: {e}")
            return True
        else:
            print(f"❌ {test_name}: Audio file generation failed")
            return False
    
    # Verify audio_files directory exists and is writable
    audio_dir = Path("audio_files")
    if not audio_dir.exists():
        print(f"📁 Creating audio_files directory: {audio_dir.absolute()}")
        audio_dir.mkdir(exist_ok=True)
    else:
        print(f"📁 Audio files directory: {audio_dir.absolute()}")
        print(f"   Currently contains {len(list(audio_dir.glob('*')))} files")
    
    print("\n2. Testing various text inputs...")
    print("-" * 40)
    
    # Test cases with different characteristics
    test_cases = [
        ("Basic test", "Hello! This is a test of the Chatterbox TTS model integration."),
        ("Short text", "Short text."),
        ("Long text", "This is a longer text example that tests how the TTS handles larger input sizes and different punctuation! Are there any issues? Let's see how well it performs with extended content that includes multiple sentences."),
        ("Numbers and symbols", "Testing numbers: 123, 4567. Special characters: @!#$%^6*()? How does it handle these?"),
        ("Mixed content", "The quick brown fox jumps over the lazy dog. 42 is the answer to life, the universe, and everything!"),
    ]
    
    successful_tests = 0
    total_tests = len(test_cases)
    
    for test_name, test_text in test_cases:
        print(f"\n🔄 {test_name}:")
        print(f"   Text: '{test_text[:80]}{'...' if len(test_text) > 80 else ''}'")
        
        try:
            audio_file = await tts_processor.text_to_speech(test_text)
            if verify_audio_file(audio_file, test_name):
                successful_tests += 1
        except Exception as e:
            print(f"❌ {test_name}: Error during TTS conversion: {e}")
    
    print("\n3. Test Summary")
    print("-" * 40)
    print(f"Tests passed: {successful_tests}/{total_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # List generated files
    generated_files = list(audio_dir.glob('*'))
    print(f"\n📁 Audio files in directory: {len(generated_files)}")
    for file in sorted(generated_files)[-5:]:  # Show last 5 files
        file_size = file.stat().st_size
        print(f"   {file.name} ({file_size:,} bytes)")

if __name__ == "__main__":
    asyncio.run(test_chatterbox_tts())
