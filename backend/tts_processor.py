"""
Text-to-Speech Processor using Chatterbox TTS with macOS say fallback
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import asyncio
import torch
import torchaudio as ta
import sys
from pathlib import Path

# Import Chatterbox from the local directory
try:
    # Add local chatterbox to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "chatterbox" / "src"))
    from chatterbox.tts import ChatterboxTTS
except ImportError as e:
    logging.error(f"Failed to import Chatterbox TTS: {e}")
    ChatterboxTTS = None

logger = logging.getLogger(__name__)

class TTSProcessor:
    """Handles text-to-speech conversion using Chatterbox TTS with macOS say fallback"""
    
    def __init__(self):
        self.model = None
        self.device = self._detect_device()
        self.temp_dir = Path(tempfile.gettempdir()) / "speech_app"
        self.temp_dir.mkdir(exist_ok=True)
        self._ready = False
        self._fallback_mode = False
    
    def _detect_device(self) -> str:
        """Detect the best available device for M3 Mac"""
        if torch.backends.mps.is_available():
            logger.info("🍎 Detected M3 Mac - using MPS (Metal Performance Shaders) for acceleration")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("🚀 CUDA available - using GPU acceleration")
            return "cuda"
        else:
            logger.info("💻 Using CPU (no hardware acceleration available)")
            return "cpu"
    
    def _setup_mps_patch(self):
        """Apply MPS-specific patches for M3 Mac compatibility (following official example)"""
        if self.device == "mps":
            # Apply the exact same patch as the official example_for_mac.py
            map_location = torch.device(self.device)
            torch_load_original = torch.load
            
            def patched_torch_load(*args, **kwargs):
                if 'map_location' not in kwargs:
                    kwargs['map_location'] = map_location
                return torch_load_original(*args, **kwargs)
            
            torch.load = patched_torch_load
            logger.info("✅ Applied MPS compatibility patches (following official Mac example)")

    async def _initialize(self):
        """Initialize the Chatterbox TTS model"""
        try:
            if ChatterboxTTS is None:
                raise ImportError("Chatterbox TTS not available")
            
            logger.info(f"🔄 Initializing Chatterbox TTS model on {self.device.upper()}...")
            
            # Apply MPS patches if needed
            self._setup_mps_patch()
            
            # Initialize Chatterbox TTS with detected device
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            
            # Verify model is properly loaded
            if self.model is not None:
                self._ready = True
                self._fallback_mode = False
                logger.info(f"✅ Chatterbox TTS model initialized successfully on {self.device.upper()}")
                if self.device == "mps":
                    logger.info("🚀 M3 Mac acceleration enabled - expect fast generation times!")
                elif self.device == "cuda":
                    logger.info("⚡ GPU acceleration enabled - expect very fast generation times!")
                else:
                    logger.warning("⚠️  Running on CPU - expect slower generation times")
            else:
                raise Exception("Model loaded but is None")
                
        except Exception as e:
            logger.error(f"❌ Chatterbox TTS model initialization failed: {e}")
            logger.warning("⚠️  Falling back to macOS say command for TTS")
            # Set model to None to ensure fallback is used
            self.model = None
            self._ready = True
            self._fallback_mode = True
    
    def is_ready(self) -> bool:
        """Check if the TTS processor is ready"""
        return self._ready
    
    async def text_to_speech(self, text: str, fast_mode: bool = False) -> Optional[Path]:
        """
        Convert text to speech and save it as an audio file
        
        Args:
            text: Text to convert
            fast_mode: If True, use macOS say for immediate response
            
        Returns:
            Path to the audio file
        """
        if not self._ready:
            logger.error("TTS model is not ready")
            return None
        
        # If fast_mode is enabled or model is not available, use fallback immediately
        if fast_mode or self._fallback_mode or self.model is None:
            if fast_mode:
                logger.info("🏃 Fast mode enabled - using macOS say for immediate response")
            else:
                logger.info("🔄 Chatterbox TTS model not available, using macOS say fallback")
            return await self._fallback_to_say(text)
            
        try:
            # Create unique filename
            filename = f"chatterbox_{abs(hash(text)) % 10000}.wav"
            audio_file = self.temp_dir / filename
            final_audio_file = Path("audio_files") / filename
            
            # Ensure audio_files directory exists
            final_audio_file.parent.mkdir(exist_ok=True)
            
            # Clean and limit text length (Chatterbox handles long text better than Dia)
            if len(text) > 500:  # More generous limit for Chatterbox
                text = text[:500] + "..."
                logger.info(f"📝 Truncated long text to: {text[:100]}...")
            else:
                logger.info(f"🎙️ Generating TTS for: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            # Generate audio using Chatterbox following the official Mac example pattern
            # Use simpler parameters for better performance
            wav_tensor = self.model.generate(text)
            
            # Save the audio using torchaudio (wav_tensor is already the right format)
            ta.save(str(final_audio_file), wav_tensor, self.model.sr)
            
            if final_audio_file.exists():
                file_size = final_audio_file.stat().st_size
                duration = wav_tensor.shape[-1] / self.model.sr
                logger.info(f"✅ Chatterbox TTS audio saved: {final_audio_file}")
                logger.info(f"📊 File size: {file_size:,} bytes, Duration: {duration:.2f}s, Sample rate: {self.model.sr} Hz")
                return final_audio_file
            
            logger.error("❌ Audio file was not created")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error with Chatterbox TTS synthesis: {e}")
            logger.info("🔄 Attempting fallback to macOS say command...")
            # Try fallback to macOS say command if Chatterbox fails
            return await self._fallback_to_say(text)
            
    async def _fallback_to_say(self, text: str) -> Optional[Path]:
        """
        Fallback to macOS say command if Chatterbox TTS fails
        
        Args:
            text: Input text
            
        Returns:
            Path to the audio file or None if failed
        """
        try:
            logger.info("🍎 Falling back to macOS say command for TTS")
            filename = f"macos_say_{abs(hash(text)) % 10000}.wav"
            audio_file = self.temp_dir / filename
            final_audio_file = Path("audio_files") / filename
            
            # Ensure audio_files directory exists
            final_audio_file.parent.mkdir(exist_ok=True)
            
            # Clean text for macOS say (remove problematic characters)
            clean_text = text.replace('"', '').replace("'", "").strip()
            if len(clean_text) > 300:  # macOS say has limitations
                clean_text = clean_text[:300] + "..."
                logger.info(f"📝 Truncated text for macOS say: {clean_text[:100]}...")
            
            logger.info(f"🎙️ Generating fallback audio for: {clean_text[:100]}{'...' if len(clean_text) > 100 else ''}")
            
            # Use macOS say command to generate AIFF file, then convert to WAV
            aiff_file = audio_file.with_suffix('.aiff')
            
            # Generate audio with say command using a better voice
            process = await asyncio.create_subprocess_exec(
                'say', '-v', 'Samantha', '-o', str(aiff_file), clean_text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and aiff_file.exists():
                aiff_size = aiff_file.stat().st_size
                logger.info(f"✅ macOS say generated AIFF file: {aiff_size:,} bytes")
                
                # Convert AIFF to WAV using ffmpeg if available
                try:
                    logger.info("🔄 Converting AIFF to WAV using ffmpeg...")
                    convert_process = await asyncio.create_subprocess_exec(
                        'ffmpeg', '-i', str(aiff_file), '-y', str(audio_file),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    convert_stdout, convert_stderr = await convert_process.communicate()
                    
                    if convert_process.returncode == 0 and audio_file.exists():
                        import shutil
                        shutil.copy(audio_file, final_audio_file)
                        aiff_file.unlink(missing_ok=True)  # Clean up AIFF file
                        audio_file.unlink(missing_ok=True)  # Clean up temp WAV file
                        
                        wav_size = final_audio_file.stat().st_size
                        logger.info(f"✅ Fallback TTS audio saved: {final_audio_file}")
                        logger.info(f"📊 File size: {wav_size:,} bytes (converted from AIFF)")
                        return final_audio_file
                    else:
                        logger.warning(f"FFmpeg conversion failed with return code {convert_process.returncode}")
                        if convert_stderr:
                            logger.warning(f"FFmpeg error: {convert_stderr.decode().strip()}")
                        raise Exception("FFmpeg conversion failed")
                        
                except Exception as ffmpeg_error:
                    logger.warning(f"⚠️ FFmpeg conversion failed: {ffmpeg_error}")
                    logger.info("📁 Keeping AIFF file as fallback format")
                    # If ffmpeg fails, move AIFF file to final location
                    final_aiff_file = final_audio_file.with_suffix('.aiff')
                    import shutil
                    shutil.copy(aiff_file, final_aiff_file)
                    aiff_file.unlink(missing_ok=True)
                    
                    aiff_size = final_aiff_file.stat().st_size
                    logger.info(f"✅ Fallback TTS audio saved (AIFF format): {final_aiff_file}")
                    logger.info(f"📊 File size: {aiff_size:,} bytes")
                    return final_aiff_file
            else:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                logger.error(f"❌ macOS say command failed with return code {process.returncode}")
                logger.error(f"Error details: {error_msg}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error with macOS say fallback: {e}")
            logger.error("💔 Both Chatterbox TTS and macOS say fallback have failed")
            return None


