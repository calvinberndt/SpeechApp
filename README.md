# Voice AI Assistant

A real-time voice AI assistant that combines Kyutai's Speech-to-Text, local Ollama LLM models, and Kyutai's Text-to-Speech into a beautiful web interface with an animated AI orb.

## Features

- 🎤 **Real-time Speech Recognition** using Kyutai STT (2.6B English model)
- 🤖 **Local LLM Integration** with Ollama (OpenHermes, Qwen, etc.)
- 🔊 **Text-to-Speech** using Kyutai TTS
- 🌟 **Beautiful AI Orb Interface** with status animations
- 💬 **Conversation Display** showing both user and AI messages
- ⚙️ **Settings Panel** for model selection and preferences
- 📱 **Responsive Design** for desktop and mobile

## Prerequisites

1. **Python 3.8+** with pip
2. **Ollama** installed and running locally
3. **moshi-mlx** package installed
4. At least one Ollama model downloaded (e.g., `openhermes` or `qwen2.5:4b`)

## Setup Instructions

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Start Ollama service
ollama serve
```

### 2. Download an LLM Model

```bash
# Download OpenHermes (recommended)
ollama pull openhermes

# Or download Qwen 2.5 4B
ollama pull qwen2.5:4b
```

### 3. Install Dependencies

```bash
cd /Users/calvinberndt/_GitHub/SpeechApp
pip install -r requirements.txt
```

### 4. Verify Kyutai Models

Make sure the Kyutai models work:

```bash
# Test STT
python -m moshi_mlx.run_inference --hf-repo kyutai/stt-2.6b-en-mlx --help

# Test TTS
python -m moshi_mlx.run_tts --help
```

## Running the Application

1. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

2. **Run the Voice AI Assistant**:
   ```bash
   cd /Users/calvinberndt/_GitHub/SpeechApp
   python main.py
   ```

3. **Open your browser** and go to:
   ```
   http://localhost:8000
   ```

## Usage

### Voice Input
1. **Hold the microphone button** and speak
2. **Release the button** when finished
3. The AI will process your speech and respond with both text and voice

### Text Input
1. **Type your message** in the text input field
2. **Click Send** or press Enter
3. The AI will respond with both text and voice

### Settings
- **Click the gear icon** in the top-right corner
- **Select your preferred AI model** (OpenHermes, Qwen, etc.)
- **Toggle voice output** on/off

## Architecture

### Backend Components
- **FastAPI Server**: Main web server with WebSocket support
- **Speech Processor**: Handles Kyutai STT integration
- **LLM Client**: Communicates with Ollama models
- **TTS Processor**: Handles Kyutai TTS integration

### Frontend
- **WebSocket Client**: Real-time communication with backend
- **AI Orb Visualization**: Animated status indicator
- **Conversation Interface**: Message display and controls
- **Audio Playback**: Plays AI-generated speech

### Models Used
- **STT**: `kyutai/stt-2.6b-en-mlx` (English speech recognition)
- **LLM**: Local Ollama models (OpenHermes, Qwen, etc.)
- **TTS**: `kyutai/tts-2.6b-en-mlx` (English speech synthesis)

## Troubleshooting

### Ollama Not Found
- Make sure Ollama is installed and running: `ollama serve`
- Check if your model is available: `ollama list`

### Audio Issues
- Check browser permissions for microphone access
- Ensure audio files are being generated in the `audio_files/` directory

### Model Loading Errors
- Verify `moshi-mlx` is properly installed
- Check if the models download correctly on first use
- Monitor console logs for detailed error messages

### Performance Tips
- Use a GPU-enabled Mac for better performance
- Consider using smaller models like `qwen2.5:4b` for faster responses
- Clear conversation history periodically to free up memory

## API Endpoints

- `GET /`: Main application interface
- `WebSocket /ws/{client_id}`: Real-time communication
- `GET /audio/{filename}`: Serve generated audio files
- `GET /health`: Health check for all services

## Development

### Project Structure
```
SpeechApp/
├── main.py                 # FastAPI application
├── backend/
│   ├── speech_processor.py # STT integration
│   ├── llm_client.py      # Ollama integration
│   └── tts_processor.py   # TTS integration
├── templates/
│   └── index.html         # Main UI template
├── static/
│   ├── style.css          # UI styles and animations
│   └── app.js            # Frontend JavaScript
├── audio_files/           # Generated audio files
└── requirements.txt       # Python dependencies
```

### Adding New Features
1. **Backend**: Modify the appropriate processor class
2. **WebSocket**: Update message handling in `main.py`
3. **Frontend**: Add UI elements and JavaScript handlers
4. **Styling**: Update CSS animations and themes

## License

This project uses various open-source components:
- FastAPI (MIT License)
- Kyutai models (CC-BY 4.0 for models, MIT/Apache for code)
- Ollama (Apache 2.0 License)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review console logs for error details
3. Verify all prerequisites are properly installed
4. Test individual components separately
