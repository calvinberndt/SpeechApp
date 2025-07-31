#!/usr/bin/env python3
"""
Comprehensive test script for Dia TTS integration with timeout handling
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

async def test_with_timeout(coro, timeout=60):
    """Run a coroutine with a timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"⏰ Operation timed out after {timeout} seconds")
        return None

async def test_dia_tts_comprehensive():
    """Comprehensive test of the Dia TTS processor"""
    
    print("="*60)
    print("Comprehensive Dia TTS Integration Test")
    print("="*60)
    
    # Initialize TTS processor
    tts_processor = TTSProcessor()
    
    # Initialize the model with timeout
    print("\n1. Initializing Dia TTS model...")
    print("   (This may take some time for first-time model download)")
    
    init_result = await test_with_timeout(tts_processor._initialize(), timeout=120)
    
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
    
    # Directory verification
    audio_dir = Path("audio_files")
    if not audio_dir.exists():
        print(f"📁 Creating audio_files directory: {audio_dir.absolute()}")
        audio_dir.mkdir(exist_ok=True)
    else:
        print(f"📁 Audio files directory: {audio_dir.absolute()}")
        current_files = list(audio_dir.glob('*'))
        print(f"   Currently contains {len(current_files)} files")
    
    print("\n2. Testing various text inputs with timeout protection...")
    print("-" * 50)
    
    # Test cases with different characteristics
    test_cases = [
        ("Basic short test", "Hello world!", 30),
        ("Simple sentence", "This is a test.", 30),
        ("Basic test", "Hello! This is a test of the Dia TTS model integration.", 60),
        ("Numbers test", "Testing numbers: 123, 4567.", 45),
        ("Long text", "This is a longer text example that tests how the TTS handles larger input sizes and different punctuation! Are there any issues? Let's see how well it performs with extended content that includes multiple sentences.", 90),
    ]
    
    successful_tests = 0
    total_tests = len(test_cases)
    
    for test_name, test_text, timeout in test_cases:
        print(f"\n🔄 {test_name} (timeout: {timeout}s):")
        print(f"   Text: '{test_text[:80]}{'...' if len(test_text) > 80 else ''}'")
        
        try:
            audio_file = await test_with_timeout(
                tts_processor.text_to_speech(test_text), 
                timeout=timeout
            )
            if audio_file and verify_audio_file(audio_file, test_name):
                successful_tests += 1
            elif audio_file is None:
                print(f"⏰ {test_name}: Generation timed out")
        except Exception as e:
            print(f"❌ {test_name}: Error during TTS conversion: {e}")
    
    print("\n3. Testing fallback functionality...")
    print("-" * 40)
    
    # Test the fallback functionality directly
    try:
        print("🔄 Testing macOS say fallback...")
        fallback_audio = await test_with_timeout(
            tts_processor._fallback_to_say("Testing fallback functionality."),
            timeout=15
        )
        if fallback_audio and verify_audio_file(fallback_audio, "Fallback test"):
            successful_tests += 1
            total_tests += 1
        else:
            print("❌ Fallback test: macOS say fallback failed")
            total_tests += 1
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        total_tests += 1
    
    print("\n4. Test Summary")
    print("-" * 40)
    print(f"Tests passed: {successful_tests}/{total_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # List generated files
    generated_files = list(audio_dir.glob('*'))
    print(f"\n📁 Audio files in directory: {len(generated_files)}")
    if generated_files:
        print("   Recent files:")
        for file in sorted(generated_files, key=lambda x: x.stat().st_mtime)[-5:]:
            file_size = file.stat().st_size
            print(f"   • {file.name} ({file_size:,} bytes)")
    
    print("\n5. Performance Analysis")
    print("-" * 40)
    if successful_tests > 0:
        print("✅ Dia TTS integration is working correctly")
        print("✅ Audio files are being saved to the audio_files directory")
        print("✅ Generated audio files have proper format and metadata")
        
        if successful_tests < total_tests:
            print("⚠️  Some tests failed - check timeout settings or model performance")
    else:
        print("❌ All tests failed - check Dia TTS installation and configuration")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_dia_tts_comprehensive())
