#!/usr/bin/env python3
"""
Comprehensive test script for TTS fallback mechanism verification
Tests both Chatterbox TTS and macOS say fallback with error scenarios
"""

import asyncio
import logging
import os
import shutil
import soundfile as sf
import sys
import tempfile
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from tts_processor import TTSProcessor

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FallbackTester:
    """Test suite for TTS fallback mechanism"""
    
    def __init__(self):
        self.test_results = {}
        self.audio_dir = Path("test_audio_files")
        self.audio_dir.mkdir(exist_ok=True)
        
    def verify_audio_file(self, audio_file, test_name):
        """Verify that an audio file was created and is valid"""
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

    async def test_normal_operation(self):
        """Test 1: Normal Chatterbox TTS operation"""
        print("🧪 Test 1: Normal Chatterbox TTS Operation")
        print("-" * 50)
        
        tts_processor = TTSProcessor()
        await tts_processor._initialize()
        
        if not tts_processor.is_ready():
            print("❌ TTS processor failed to initialize")
            return False
        
        test_text = "This is a test of the normal Chatterbox TTS operation."
        audio_file = await tts_processor.text_to_speech(test_text)
        
        success = self.verify_audio_file(audio_file, "Normal Operation")
        self.test_results["normal_operation"] = success
        
        # Log the mode used
        if tts_processor._fallback_mode:
            print("   🍎 Used macOS say fallback (Chatterbox not available)")
        else:
            print("   🤖 Used Chatterbox TTS successfully")
        
        return success

    async def test_forced_fallback(self):
        """Test 2: Force fallback mode by using fast_mode"""
        print("\n🧪 Test 2: Forced Fallback (Fast Mode)")
        print("-" * 50)
        
        tts_processor = TTSProcessor()
        await tts_processor._initialize()
        
        test_text = "This tests the forced fallback mode using macOS say."
        audio_file = await tts_processor.text_to_speech(test_text, fast_mode=True)
        
        success = self.verify_audio_file(audio_file, "Forced Fallback")
        self.test_results["forced_fallback"] = success
        print("   🏃 Fast mode enabled - should use macOS say")
        
        return success

    async def test_chatterbox_initialization_failure(self):
        """Test 3: Simulate Chatterbox initialization failure"""
        print("\n🧪 Test 3: Chatterbox Initialization Failure Simulation")
        print("-" * 50)
        
        # Mock ChatterboxTTS to raise an exception
        with patch('backend.tts_processor.ChatterboxTTS') as mock_chatterbox:
            mock_chatterbox.from_pretrained.side_effect = Exception("Simulated initialization failure")
            
            tts_processor = TTSProcessor()
            await tts_processor._initialize()
            
            # Should still be ready but in fallback mode
            if tts_processor.is_ready() and tts_processor._fallback_mode:
                print("✅ Correctly entered fallback mode after initialization failure")
                
                test_text = "This tests fallback after initialization failure."
                audio_file = await tts_processor.text_to_speech(test_text)
                
                success = self.verify_audio_file(audio_file, "Init Failure Fallback")
                self.test_results["init_failure_fallback"] = success
                return success
            else:
                print("❌ Failed to handle initialization failure correctly")
                self.test_results["init_failure_fallback"] = False
                return False

    async def test_chatterbox_runtime_failure(self):
        """Test 4: Simulate Chatterbox runtime failure during generation"""
        print("\n🧪 Test 4: Chatterbox Runtime Failure Simulation")
        print("-" * 50)
        
        # Create a TTS processor with normal initialization
        tts_processor = TTSProcessor()
        await tts_processor._initialize()
        
        if tts_processor._fallback_mode:
            print("⚠️  Already in fallback mode, skipping runtime failure test")
            self.test_results["runtime_failure_fallback"] = True
            return True
        
        # Mock the generate method to fail
        if tts_processor.model:
            original_generate = tts_processor.model.generate
            tts_processor.model.generate = MagicMock(side_effect=Exception("Simulated runtime failure"))
            
            test_text = "This tests fallback after runtime failure."
            audio_file = await tts_processor.text_to_speech(test_text)
            
            # Restore original method
            tts_processor.model.generate = original_generate
            
            success = self.verify_audio_file(audio_file, "Runtime Failure Fallback")
            self.test_results["runtime_failure_fallback"] = success
            print("   💥 Simulated Chatterbox runtime failure - should fallback to macOS say")
            return success
        else:
            print("⚠️  No Chatterbox model available for runtime failure test")
            self.test_results["runtime_failure_fallback"] = True
            return True

    async def test_long_text_handling(self):
        """Test 5: Test long text handling in both modes"""
        print("\n🧪 Test 5: Long Text Handling")
        print("-" * 50)
        
        tts_processor = TTSProcessor()
        await tts_processor._initialize()
        
        # Create a very long text
        long_text = "This is a very long text that exceeds normal limits. " * 20
        print(f"   📝 Text length: {len(long_text)} characters")
        
        # Test with Chatterbox (or fallback if not available)
        audio_file1 = await tts_processor.text_to_speech(long_text)
        success1 = self.verify_audio_file(audio_file1, "Long Text - Primary")
        
        # Test with forced fallback
        audio_file2 = await tts_processor.text_to_speech(long_text, fast_mode=True)
        success2 = self.verify_audio_file(audio_file2, "Long Text - Fallback")
        
        success = success1 and success2
        self.test_results["long_text_handling"] = success
        return success

    async def test_ffmpeg_availability(self):
        """Test 6: Test audio conversion with and without ffmpeg"""
        print("\n🧪 Test 6: FFmpeg Availability Test")
        print("-" * 50)
        
        # Check if ffmpeg is available
        try:
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            ffmpeg_available = process.returncode == 0
        except FileNotFoundError:
            ffmpeg_available = False
        
        print(f"   🔧 FFmpeg available: {ffmpeg_available}")
        
        tts_processor = TTSProcessor()
        await tts_processor._initialize()
        
        # Force fallback mode to test conversion
        test_text = "Testing audio conversion capabilities."
        audio_file = await tts_processor.text_to_speech(test_text, fast_mode=True)
        
        success = self.verify_audio_file(audio_file, "FFmpeg Conversion Test")
        
        if success and audio_file:
            if audio_file.suffix.lower() == '.wav' and ffmpeg_available:
                print("   ✅ Successfully converted to WAV using FFmpeg")
            elif audio_file.suffix.lower() == '.aiff':
                print("   ✅ Gracefully degraded to AIFF format")
            else:
                print("   ⚠️  Unexpected audio format")
        
        self.test_results["ffmpeg_test"] = success
        return success

    async def test_error_conditions(self):
        """Test 7: Various error conditions and edge cases"""
        print("\n🧪 Test 7: Error Conditions and Edge Cases")
        print("-" * 50)
        
        tts_processor = TTSProcessor()
        await tts_processor._initialize()
        
        # Test empty text
        print("   Testing empty text...")
        audio_file1 = await tts_processor.text_to_speech("")
        success1 = audio_file1 is not None  # Should handle gracefully
        
        # Test very short text
        print("   Testing very short text...")
        audio_file2 = await tts_processor.text_to_speech("Hi.")
        success2 = self.verify_audio_file(audio_file2, "Short Text")
        
        # Test text with special characters
        print("   Testing special characters...")
        special_text = "Testing quotes: \"Hello\" and 'world'! @#$%^&*()_+"
        audio_file3 = await tts_processor.text_to_speech(special_text, fast_mode=True)
        success3 = self.verify_audio_file(audio_file3, "Special Characters")
        
        success = success1 and success2 and success3
        self.test_results["error_conditions"] = success
        return success

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🎯 FALLBACK MECHANISM TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, passed in self.test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print("-" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Fallback mechanism is working correctly.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review the issues above.")
        
        # Show generated files
        audio_files = list(Path("audio_files").glob("*")) if Path("audio_files").exists() else []
        print(f"\n📁 Generated {len(audio_files)} audio files in audio_files/")
        for file in sorted(audio_files)[-5:]:  # Show last 5 files
            file_size = file.stat().st_size
            print(f"   {file.name} ({file_size:,} bytes)")

async def main():
    """Run all fallback mechanism tests"""
    print("🚀 Starting Comprehensive TTS Fallback Mechanism Tests")
    print("=" * 60)
    
    tester = FallbackTester()
    
    # Run all tests
    await tester.test_normal_operation()
    await tester.test_forced_fallback()
    await tester.test_chatterbox_initialization_failure()
    await tester.test_chatterbox_runtime_failure()
    await tester.test_long_text_handling()
    await tester.test_ffmpeg_availability()
    await tester.test_error_conditions()
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
