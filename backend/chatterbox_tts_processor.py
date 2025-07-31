"""
Chatterbox-TTS Processor optimized for Apple Silicon with performance enhancements
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional, List

import torch
import torchaudio
from chatterbox import ChatterboxTTS

logger = logging.getLogger(__name__)

class ChatterboxTTSProcessor:
    """Handles text-to-speech conversion using Chatterbox-TTS optimized for Apple Silicon"""

    def __init__(self):
        self.model = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.map_location = torch.device(self.device)
        self.temp_dir = Path(tempfile.gettempdir()) / "speech_app"
        self.temp_dir.mkdir(exist_ok=True)
        self._ready = False
        
        # Apply torch.load monkey patch for Apple Silicon compatibility
        if not hasattr(torch, '_original_load'):
            torch._original_load = torch.load
            def patched_torch_load(*args, **kwargs):
                if 'map_location' not in kwargs:
                    kwargs['map_location'] = self.map_location
                return torch._original_load(*args, **kwargs)
            torch.load = patched_torch_load
            logger.info(f"Applied torch.load patch for {self.device} compatibility")

    async def _initialize(self):
        """Initialize the Chatterbox TTS model with performance optimizations"""
        try:
            logger.info(f"Initializing Chatterbox TTS model on {self.device}...")
            
            # Performance optimization: Set attention implementation via environment
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
            # Set attention implementation for transformers models
            os.environ["TRANSFORMERS_ATTENTION_TYPE"] = "eager"
            
            # Load the model with performance optimizations
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            
            # Performance optimizations
            if self.device != "cpu":
                logger.info(f"Applying performance optimizations for {self.device}...")
                
                # Move model components to the target device
                if hasattr(self.model, 't3'):
                    self.model.t3 = self.model.t3.to(self.device)
                
                if hasattr(self.model, 's3gen'):
                    self.model.s3gen = self.model.s3gen.to(self.device)
                
                if hasattr(self.model, 've'):
                    self.model.ve = self.model.ve.to(self.device)
                
                logger.info("Moved all model components to MPS device (without half precision to avoid type conflicts)")
            
            # Set models to eval mode for inference optimization
            if hasattr(self.model, 't3'):
                self.model.t3.eval()
            if hasattr(self.model, 's3gen'):
                self.model.s3gen.eval()
            if hasattr(self.model, 've'):
                self.model.ve.eval()
            
            # Try to compile models for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device != "cpu":
                try:
                    if hasattr(self.model, 't3'):
                        self.model.t3 = torch.compile(self.model.t3, mode="reduce-overhead")
                        logger.info("Applied torch.compile to T3 model")
                    if hasattr(self.model, 's3gen'):
                        self.model.s3gen = torch.compile(self.model.s3gen, mode="reduce-overhead")
                        logger.info("Applied torch.compile to S3Gen model")
                except Exception as e:
                    logger.warning(f"torch.compile optimization failed: {e}")
            
            # AGGRESSIVE SPEED OPTIMIZATION: Monkey patch T3 inference for faster generation
            if hasattr(self.model, 't3') and hasattr(self.model.t3, 'inference'):
                original_inference = self.model.t3.inference
                def faster_inference(*args, **kwargs):
                    # Reduce max_new_tokens for faster generation in fast mode
                    if 'max_new_tokens' in kwargs:
                        original_tokens = kwargs['max_new_tokens']
                        kwargs['max_new_tokens'] = min(600, original_tokens)  # Cap at 600 for speed
                        logger.debug(f"Reduced max_new_tokens from {original_tokens} to {kwargs['max_new_tokens']}")
                    return original_inference(*args, **kwargs)
                
                self.model.t3.inference = faster_inference
                logger.info("Applied aggressive speed optimization to T3 inference")
            
            self._ready = True
            logger.info(f"✅ Chatterbox TTS model initialized successfully on {self.device}")

        except Exception as e:
            logger.error(f"❌ Chatterbox TTS model initialization failed: {e}")
            logger.warning(f"⚠️  Falling back to CPU mode")
            try:
                # Fallback to CPU
                self.device = "cpu"
                self.map_location = torch.device(self.device)
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                self._ready = True
                logger.info("✅ Chatterbox TTS model initialized on CPU fallback")
            except Exception as fallback_error:
                logger.error(f"❌ CPU fallback also failed: {fallback_error}")
                self._ready = False

    def is_ready(self) -> bool:
        """Check if the TTS processor is ready"""
        return self._ready

    def _split_text_into_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences for parallel processing"""
        import re
        
        # Split by sentences (including multiple punctuation)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # If no sentence breaks found but text is long, split by commas or every ~100 chars
        if len(sentences) == 1 and len(text) > 100:
            # Try splitting by commas first
            comma_splits = re.split(r',\s+', text)
            if len(comma_splits) > 1:
                sentences = comma_splits
            else:
                # Split every ~80 characters at word boundaries
                words = text.split()
                current_chunk = ""
                sentences = []
                for word in words:
                    if len(current_chunk) + len(word) > 80 and current_chunk:
                        sentences.append(current_chunk.strip())
                        current_chunk = word
                    else:
                        current_chunk += (" " + word if current_chunk else word)
                if current_chunk:
                    sentences.append(current_chunk.strip())
        
        # Clean up and filter out empty sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short fragments
                cleaned_sentences.append(sentence)
        
        logger.info(f"Split text into {len(cleaned_sentences)} chunks for streaming processing")
        return cleaned_sentences if cleaned_sentences else [text]

    async def _generate_chunk_audio(self, text: str, audio_prompt_path: Optional[str] = None,
                                  exaggeration: float = 1.0, cfg_weight: float = 0.3, 
                                  fast_mode: bool = True) -> Optional[torch.Tensor]:
        """Generate audio for a single text chunk"""
        try:
            # Performance optimizations for generation
            with torch.inference_mode():  # Disable gradient computation for faster inference
                # Generate audio using Chatterbox TTS with enhanced parameters
                generation_kwargs = {
                    "text": text,
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight
                }
                
                # Fast mode optimizations
                if fast_mode:
                    # Only override with supported parameters for faster synthesis
                    generation_kwargs["cfg_weight"] = 0.2  # Lower CFG for faster generation
                    generation_kwargs["temperature"] = 0.6  # Lower temperature for faster sampling
                    generation_kwargs["min_p"] = 0.1  # Higher min_p for faster sampling
                    generation_kwargs["top_p"] = 0.8  # Lower top_p for faster sampling
                    generation_kwargs["repetition_penalty"] = 1.1  # Lower penalty for speed
                    logger.info(f"Applied fast mode parameters: {list(generation_kwargs.keys())}")
                
                # Add audio prompt if provided
                if audio_prompt_path and Path(audio_prompt_path).exists():
                    generation_kwargs["audio_prompt_path"] = audio_prompt_path
                    logger.info(f"Using audio prompt: {audio_prompt_path}")
                
                logger.info(f"Calling ChatterboxTTS.generate with parameters: {list(generation_kwargs.keys())}")
                return self.model.generate(**generation_kwargs)
        
        except Exception as e:
            logger.error(f"Error generating audio chunk: {e}")
            return None

    async def text_to_speech(self, text: str, audio_prompt_path: Optional[str] = None, 
                           exaggeration: float = 1.0, cfg_weight: float = 0.3,
                           fast_mode: bool = True, use_streaming: bool = True) -> Optional[Path]:
        """
        Convert text to speech and save it as an audio file

        Args:
            text: Text to convert
            audio_prompt_path: Optional path to audio file for voice cloning
            exaggeration: Voice exaggeration factor (default: 1.0)
            cfg_weight: Classifier-free guidance weight (default: 0.3)
            fast_mode: Enable fast generation mode (default: True)
            use_streaming: Enable streaming TTS (process sentences as ready) (default: True)

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
            
            # Check if we should use sentence-based streaming (for any text > 30 chars)
            if use_streaming and (len(text) > 30 or '.' in text or '!' in text or '?' in text):
                sentences = self._split_text_into_sentences(text)
                
                if len(sentences) > 1:
                    logger.info(f"🚀 STREAMING MODE: Processing first sentence immediately, others in parallel")
                    
                    # STREAMING APPROACH: Process first sentence immediately for faster response
                    first_sentence = sentences[0]
                    remaining_sentences = sentences[1:]
                    
                    logger.info(f"⚡ Processing first sentence: '{first_sentence[:50]}...'")
                    first_audio = await self._generate_chunk_audio(
                        first_sentence, audio_prompt_path, exaggeration, cfg_weight, fast_mode
                    )
                    
                    if first_audio is None:
                        logger.error("Failed to generate audio for first sentence")
                        return None
                    
                    if remaining_sentences:
                        logger.info(f"🔄 Processing {len(remaining_sentences)} remaining sentences in parallel")
                        
                        # Process remaining sentences in parallel
                        remaining_tasks = [
                            self._generate_chunk_audio(
                                sentence, audio_prompt_path, exaggeration, cfg_weight, fast_mode
                            ) for sentence in remaining_sentences
                        ]
                        
                        # Wait for remaining sentences
                        remaining_chunks = await asyncio.gather(*remaining_tasks, return_exceptions=True)
                        
                        # Filter valid chunks
                        valid_remaining = []
                        for i, chunk in enumerate(remaining_chunks):
                            if isinstance(chunk, Exception):
                                logger.warning(f"Failed to generate sentence {i+2}: {chunk}")
                            elif chunk is not None:
                                valid_remaining.append(chunk)
                        
                        # Combine first sentence with remaining
                        if valid_remaining:
                            silence_samples = int(0.15 * 24000)  # 0.15s pause between sentences
                            silence = torch.zeros(1, silence_samples)
                            
                            all_chunks = [first_audio]
                            for chunk in valid_remaining:
                                all_chunks.extend([silence, chunk])
                            
                            audio_data = torch.cat(all_chunks, dim=1)
                            logger.info(f"✅ FAST: Combined {1 + len(valid_remaining)} sentences with natural pauses")
                        else:
                            logger.warning("Only first sentence succeeded, using that")
                            audio_data = first_audio
                    else:
                        audio_data = first_audio
                else:
                    # Single sentence, process normally
                    audio_data = await self._generate_chunk_audio(
                        text, audio_prompt_path, exaggeration, cfg_weight, fast_mode
                    )
            else:
                # Process as single text (no punctuation or streaming disabled)
                audio_data = await self._generate_chunk_audio(
                    text, audio_prompt_path, exaggeration, cfg_weight, fast_mode
                )
            
            # Check if audio generation was successful
            if audio_data is None:
                logger.error("Audio generation failed")
                return None
            
            # Save audio tensor to file
            # ChatterboxTTS returns a torch.Tensor with shape [1, samples]
            if isinstance(audio_data, torch.Tensor):
                # Ensure the tensor is on CPU for saving
                audio_tensor = audio_data.cpu()
                
                # Use the model's sample rate if available, otherwise default to 24kHz
                sample_rate = getattr(self.model, 'sr', 24000)
                
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

