# ThinkChain-Assistant
A Flask-based AI virtual assistant that solves problems via a chain of agents, with optional web research and voice output. Powered by Ollama, it streams insights in real-time.

## Features
- **Multi-Agent Workflow**: Processes queries through researcher, investigator, analyst, strategist, and advisor agents.
- **Optional Web Research**: Uses DuckDuckGo, with fallback to model-based responses if no suitable results.
- **Real-Time Streaming**: Displays output in the web UI as it’s generated.
- **Voice Output**: Converts responses to speech using the Web Speech API.
- **Ollama Integration**: Supports models like Mistral and Llama3.

## Prerequisites
- Python 3.10+
- Ollama server running locally on `http://127.0.0.1:11434`
- Modern browser (e.g., Chrome, Edge) with Web Speech API support
