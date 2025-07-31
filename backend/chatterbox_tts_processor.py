"""
Chatterbox-TTS Processor optimized for Apple Silicon
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from chatterbox import ChatterboxTTS

logger = logging.getLogger(__name__)

class ChatterboxTTSProcessor:
    """Handles text-to-speech conversion using Chatterbox-TTS optimized for Apple Silicon"""

    def __init__(self):
        self.model = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.temp_dir = Path(tempfile.gettempdir()) / "speech_app"
        self.temp_dir.mkdir(exist_ok=True)
        self._ready = False

    async def _initialize(self):
        """Initialize the Chatterbox TTS model"""
        try:
            logger.info(f"Initializing Chatterbox TTS model on {self.device}...")
            
            # Load the model - it will download automatically if needed
            self.model = ChatterboxTTS.from_pretrained(self.device)
            
            # Move model components to the target device if not CPU
            if self.device != "cpu":
                logger.info(f"Moving model components to {self.device}...")
                if hasattr(self.model, 't3'):
                    self.model.t3 = self.model.t3.to(self.device)
                if hasattr(self.model, 's3gen'):
                    self.model.s3gen = self.model.s3gen.to(self.device)
                if hasattr(self.model, 've'):
                    self.model.ve = self.model.ve.to(self.device)
            
            self._ready = True
            logger.info(f"✅ Chatterbox TTS model initialized successfully on {self.device}")

        except Exception as e:
            logger.error(f"❌ Chatterbox TTS model initialization failed: {e}")
            logger.warning(f"⚠️  Falling back to CPU mode")
            try:
                # Fallback to CPU
                self.device = "cpu"
                self.model = ChatterboxTTS.from_pretrained(self.device)
                self._ready = True
                logger.info("✅ Chatterbox TTS model initialized on CPU fallback")
            except Exception as fallback_error:
                logger.error(f"❌ CPU fallback also failed: {fallback_error}")
                self._ready = False

    def is_ready(self) -> bool:
        """Check if the TTS processor is ready"""
        return self._ready

    async def text_to_speech(self, text: str) -> Optional[Path]:
        """
        Convert text to speech and save it as an audio file

        Args:
            text: Text to convert

        Returns:
            Path to the audio file
        """
        if not self._ready:
            logger.error("TTS model is not ready")
            return None

        try:
            # Create unique filename
            filename = f"chatterbox_{abs(hash(text)) % 10000}.wav"
            temp_audio_file = self.temp_dir / filename
            final_audio_file = Path("audio_files") / filename
            
            # Ensure audio_files directory exists
            final_audio_file.parent.mkdir(exist_ok=True)
            
            logger.info(f"Generating TTS for: {text[:100]}...")
            
            # Generate audio using Chatterbox TTS
            audio_data = self.model.generate(text=text)
            
            # Save audio tensor to file
            # ChatterboxTTS returns a torch.Tensor with shape [1, samples]
            if isinstance(audio_data, torch.Tensor):
                # Ensure the tensor is on CPU for saving
                audio_tensor = audio_data.cpu()
                
                # ChatterboxTTS typically uses 24kHz sample rate
                sample_rate = 24000
                
                # Save as WAV file
                torchaudio.save(str(temp_audio_file), audio_tensor, sample_rate)
                logger.info(f"Audio tensor shape: {audio_tensor.shape}, sample_rate: {sample_rate}")
            else:
                logger.error(f"Unexpected audio data type: {type(audio_data)}")
                return None
            
            # Move to final location
            if temp_audio_file.exists():
                import shutil
                shutil.copy(temp_audio_file, final_audio_file)
                temp_audio_file.unlink(missing_ok=True)  # Clean up temp file
                logger.info(f"✅ TTS audio saved to {final_audio_file}")
                return final_audio_file
            else:
                logger.error("Audio file was not created")
                return None

        except Exception as e:
            logger.error(f"❌ Error synthesizing speech: {e}")
            return None

