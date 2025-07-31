"""
Text-to-Speech Processor using Dia TTS
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import asyncio
import aiofiles
import soundfile as sf
import sys
from pathlib import Path

# Import Dia from the correct path
try:
    # Try local dia directory first
    sys.path.insert(0, str(Path(__file__).parent.parent / "dia"))
    from dia.model import Dia
except ImportError:
    try:
        # Fallback to installed dia package
        from dia.model import Dia
    except ImportError:
        # Last resort - try alternative import paths
        try:
            from dia.dia import Dia
        except ImportError:
            from dia import Dia

logger = logging.getLogger(__name__)

class TTSProcessor:
    """Handles text-to-speech conversion using Dia TTS"""
    
    def __init__(self, model_name: str = "nari-labs/Dia-1.6B-0626"):
        self.model_name = model_name
        self.model = None
        self.temp_dir = Path(tempfile.gettempdir()) / "speech_app"
        self.temp_dir.mkdir(exist_ok=True)
        self._ready = False

    async def _initialize(self):
        """Initialize the Dia TTS model"""
        try:
            logger.info("Initializing Dia TTS model...")
            # Load the Dia model with optimized settings for CPU
            self.model = Dia.from_pretrained(
                self.model_name, 
                compute_dtype="float16"  # Keep float16 for memory efficiency
            )
            
            # Verify model is properly loaded
            if self.model is not None:
                self._ready = True
                logger.info("✅ Dia TTS model initialized successfully")
                logger.warning("⚠️  Running on CPU - expect slower generation times (2-3 minutes per request)")
            else:
                raise Exception("Model loaded but is None")
                
        except Exception as e:
            logger.error(f"❌ Dia TTS model initialization failed: {e}")
            logger.warning("⚠️  Falling back to macOS say command for TTS")
            # Set model to None to ensure fallback is used
            self.model = None
            self._ready = True
    
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
        
        # If fast_mode is enabled or Dia model is not available, use fallback immediately
        if fast_mode or self.model is None:
            if fast_mode:
                logger.info("Fast mode enabled - using macOS say for immediate response")
            else:
                logger.info("Dia TTS model not available, using fallback")
            return await self._fallback_to_say(text)
            
        try:
            # Create unique filename
            filename = f"output_{abs(hash(text)) % 10000}.wav"
            audio_file = self.temp_dir / filename
            final_audio_file = Path("audio_files") / filename
            
            # Ensure audio_files directory exists
            final_audio_file.parent.mkdir(exist_ok=True)
            
            # Format text for Dia TTS (use [S1] speaker tag as recommended)
            formatted_text = f"[S1] {text}"
            
            # Limit text length for faster generation (keep under 20s of audio as per guidelines)
            if len(formatted_text) > 200:  # Rough estimate for reasonable length
                formatted_text = formatted_text[:200] + "..."
                logger.info(f"Truncated long text to: {formatted_text}")
            else:
                logger.info(f"Generating TTS for: {formatted_text[:100]}...")
            
            # Generate audio using optimized parameters for CPU performance
            # Based on official examples with adjustments for speed
            audio_data = self.model.generate(
                formatted_text,  # Pass text as positional argument (like in examples)
                use_torch_compile=False,  # Keep disabled for stability
                verbose=False,  # Reduce logging overhead
                cfg_scale=3.0,  # Keep default from examples
                temperature=1.5,  # Slightly lower than example for faster convergence
                top_p=0.90,  # Match example
                cfg_filter_top_k=50,  # Match example
                # Note: max_tokens removed - let model decide
            )

            # Save the audio file directly (audio_data should be the right format)
            # No need to wrap in list - the example shows direct usage
            self.model.save_audio(str(audio_file), audio_data)
            
            if audio_file.exists():
                import shutil
                shutil.copy(audio_file, final_audio_file)
                audio_file.unlink(missing_ok=True)  # Clean up temp file
                logger.info(f"TTS audio saved to {final_audio_file}")
                return final_audio_file
            
            logger.error("Audio file was not created")
            return None
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            # Try fallback to macOS say command if Dia fails
            return await self._fallback_to_say(text)
            
    async def _fallback_to_say(self, text: str) -> Optional[Path]:
        """
        Fallback to macOS say command if Dia TTS fails
        
        Args:
            text: Input text
            
        Returns:
            Path to the audio file or None if failed
        """
        try:
            logger.info("Falling back to macOS say command for TTS")
            filename = f"output_{abs(hash(text)) % 10000}.wav"
            audio_file = self.temp_dir / filename
            final_audio_file = Path("audio_files") / filename
            
            # Ensure audio_files directory exists
            final_audio_file.parent.mkdir(exist_ok=True)
            
            # Use macOS say command to generate AIFF file, then convert to WAV
            aiff_file = audio_file.with_suffix('.aiff')
            
            # Generate audio with say command
            process = await asyncio.create_subprocess_exec(
                'say', '-o', str(aiff_file), text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and aiff_file.exists():
                # Convert AIFF to WAV using ffmpeg if available
                try:
                    convert_process = await asyncio.create_subprocess_exec(
                        'ffmpeg', '-i', str(aiff_file), '-y', str(audio_file),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await convert_process.communicate()
                    
                    if convert_process.returncode == 0 and audio_file.exists():
                        import shutil
                        shutil.copy(audio_file, final_audio_file)
                        aiff_file.unlink(missing_ok=True)  # Clean up AIFF file
                        audio_file.unlink(missing_ok=True)  # Clean up temp WAV file
                        return final_audio_file
                except Exception as ffmpeg_error:
                    logger.warning(f"FFmpeg conversion failed: {ffmpeg_error}")
                    # If ffmpeg fails, just rename AIFF to WAV
                    aiff_file.rename(final_audio_file.with_suffix('.aiff'))
                    return final_audio_file.with_suffix('.aiff')
            
            logger.error("macOS say command failed")
            return None
            
        except Exception as e:
            logger.error(f"Error with macOS say fallback: {e}")
            return None

    async def _generate_audio_with_say(self, text: str, audio_file: Path) -> bool:
        """
        Generate audio using macOS say command as fallback (removed, not required)
        """
        logger.warning("macOS say command fallback is deprecated and not available.")
        return False

