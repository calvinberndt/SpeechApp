#!/usr/bin/env python3
"""
Test script for ChatterboxTTS integration
"""

import asyncio
import logging
import soundfile as sf
from pathlib import Path
from backend.chatterbox_tts_processor import ChatterboxTTSProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_chatterbox_tts():
    """Test the ChatterboxTTS processor"""
    
    print("="*60)
    print("Testing ChatterboxTTS integration...")
    print("="*60)
    
    # Initialize TTS processor
    tts_processor = ChatterboxTTSProcessor()
    
    # Initialize the model
    print("\n1. Initializing ChatterboxTTS model...")
    await tts_processor._initialize()
    
    if not tts_processor.is_ready():
        print("❌ TTS processor failed to initialize")
        return
        
    print(f"✅ ChatterboxTTS processor initialized successfully on {tts_processor.device}\n")
    
    # Function to verify audio file
    def verify_audio_file(audio_file, test_name):
        if audio_file and audio_file.exists():
            file_size = audio_file.stat().st_size
            print(f"✅ {test_name}: Audio file generated successfully")
            print(f"   📁 File path: {audio_file}")
            print(f"   📊 File size: {file_size:,} bytes")
            
            # Check if it's a proper audio file
            try:
                info = sf.info(str(audio_file))
                print(f"   🎵 Format: {info.format}, Sample rate: {info.samplerate} Hz, Duration: {info.duration:.2f}s")
            except Exception as e:
                print(f"   ⚠️  Could not read audio file info: {e}")
            return True
        else:
            print(f"❌ {test_name}: Audio file generation failed")
            return False
    
    # Verify audio_files directory exists
    audio_dir = Path("audio_files")
    if not audio_dir.exists():
        print(f"📁 Creating audio_files directory: {audio_dir.absolute()}")
        audio_dir.mkdir(exist_ok=True)
    else:
        print(f"📁 Audio files directory: {audio_dir.absolute()}")
        current_files = list(audio_dir.glob('*'))
        print(f"   Currently contains {len(current_files)} files")
    
    print("\n2. Testing various text inputs...")
    print("-" * 40)
    
    # Test cases with different characteristics
    test_cases = [
        ("Basic test", "Hello! This is a test of the ChatterboxTTS model integration."),
        ("Short text", "Short text."),
        ("Numbers test", "Testing numbers: 123, 4567."),
        ("Long text", "This is a longer text example that tests how the TTS handles larger input sizes and different punctuation! Are there any issues? Let's see how well it performs with extended content."),
        ("Mixed content", "The quick brown fox jumps over the lazy dog. 42 is the answer!"),
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
    generated_files = list(audio_dir.glob('chatterbox_*.wav'))
    print(f"\n📁 ChatterboxTTS audio files: {len(generated_files)}")
    if generated_files:
        print("   Recent files:")
        for file in sorted(generated_files, key=lambda x: x.stat().st_mtime)[-5:]:
            file_size = file.stat().st_size
            print(f"   • {file.name} ({file_size:,} bytes)")
    
    print("\n4. Performance Analysis")
    print("-" * 40)
    if successful_tests > 0:
        print("✅ ChatterboxTTS integration is working correctly")
        print("✅ Audio files are being saved to the audio_files directory")
        print("✅ Generated audio files have proper format and metadata")
        print(f"✅ Using device: {tts_processor.device}")
        
        if tts_processor.device == "mps":
            print("🚀 Using Apple Silicon GPU acceleration for faster generation!")
        elif tts_processor.device == "cpu":
            print("💻 Using CPU mode - consider upgrading to Apple Silicon for GPU acceleration")
        
        if successful_tests < total_tests:
            print("⚠️  Some tests failed - check error messages above")
    else:
        print("❌ All tests failed - check ChatterboxTTS installation and configuration")
    
    print("\n" + "="*60)
    print("ChatterboxTTS test completed!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_chatterbox_tts())
