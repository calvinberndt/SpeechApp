"""
Speech-to-Text Processor using Kyutai STT
"""

import asyncio
import base64
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """Handles speech-to-text conversion using Kyutai STT"""
    
    def __init__(self):
        self.model_repo = "kyutai/stt-2.6b-en-mlx"
        self.temp_dir = Path(tempfile.gettempdir()) / "speech_app"
        self.temp_dir.mkdir(exist_ok=True)
        self._ready = False
    
    async def _initialize(self):
        """Initialize the STT model"""
        try:
            logger.info("Initializing Kyutai STT model...")
            # Test the model by running a quick check
            result = await self._run_stt_command(["--help"])
            if result and "usage:" in result.lower():
                self._ready = True
                logger.info("Kyutai STT model initialized successfully")
            else:
                logger.error("Failed to initialize Kyutai STT model")
        except Exception as e:
            logger.error(f"Error initializing STT model: {e}")
    
    def is_ready(self) -> bool:
        """Check if the STT processor is ready"""
        return self._ready
    
    async def process_audio(self, audio_data: str) -> str:
        """
        Process base64 encoded audio data and return transcribed text
        
        Args:
            audio_data: Base64 encoded audio data
            
        Returns:
            Transcribed text
        """
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Create temporary audio file
            audio_file = self.temp_dir / f"input_{hash(audio_data) % 10000}.wav"
            audio_file.write_bytes(audio_bytes)
            
            # Run STT
            transcription = await self._transcribe_audio(audio_file)
            
            # Clean up temporary file
            audio_file.unlink(missing_ok=True)
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return ""
    
    async def process_audio_file(self, file_path: Path) -> str:
        """
        Process an audio file and return transcribed text
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            return await self._transcribe_audio(file_path)
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {e}")
            return ""
    
    async def _transcribe_audio(self, audio_file: Path) -> str:
        """
        Run Kyutai STT on an audio file
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Run the Kyutai STT command
            result = await self._run_stt_command([str(audio_file)])
            
            if result:
                # Extract the transcription from the output
                # The model outputs info lines and then the transcription
                lines = result.split('\n')
                
                # Find the transcription (usually after "Info: steps:" line)
                transcription = ""
                capture_next = False
                
                for line in lines:
                    if "Info: steps:" in line:
                        capture_next = True
                        continue
                    elif capture_next and line.strip():
                        # This should be the transcription
                        transcription = line.strip()
                        break
                
                return transcription
            
            return ""
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
    
    async def _run_stt_command(self, args: list) -> Optional[str]:
        """
        Run the Kyutai STT command asynchronously
        
        Args:
            args: Command arguments
            
        Returns:
            Command output or None if failed
        """
        try:
            cmd = [
                "/opt/anaconda3/bin/python", "-m", "moshi_mlx.run_inference",
                "--hf-repo", self.model_repo
            ] + args
            
            # Run the command asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return stdout.decode('utf-8')
            else:
                logger.error(f"STT command failed: {stderr.decode('utf-8')}")
                return None
                
        except Exception as e:
            logger.error(f"Error running STT command: {e}")
            return None
    
    async def convert_audio_format(self, input_file: Path, output_file: Path, 
                                  sample_rate: int = 24000) -> bool:
        """
        Convert audio to the format expected by Kyutai STT (24kHz WAV)
        
        Args:
            input_file: Input audio file
            output_file: Output audio file
            sample_rate: Target sample rate (default: 24000)
            
        Returns:
            True if conversion successful
        """
        try:
            # Read audio file
            data, original_sr = sf.read(input_file)
            
            # Resample if necessary
            if original_sr != sample_rate:
                import scipy.signal
                # Calculate number of samples after resampling
                num_samples = int(len(data) * sample_rate / original_sr)
                data = scipy.signal.resample(data, num_samples)
            
            # Ensure mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Write output file
            sf.write(output_file, data, sample_rate)
            
            return True
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            return False
