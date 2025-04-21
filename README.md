# Browser Use MCP Server

A FastMCP server that allows you to automate browser tasks with natural language.

This server enables Language Models to browse the web, fill out forms, click buttons, and perform other web-based tasks via a simple API.

## Supported LLM Providers

The following LLM providers are supported for browser automation:

| Provider | Models | Tool Calling | Structured Output | JSON Mode |
|----------|--------|-------------|-------------------|-----------|
| OpenAI | GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo | ✅ | ✅ | ✅ |
| Anthropic | Claude 3 Opus, Sonnet, Haiku | ✅ | ✅ | ✅ |
| Google | Gemini Pro, Gemini Flash | ✅ | ✅ | ✅ |
| Cohere | Command R, Command R+ | ✅ | ✅ | ❌ |
| Mistral AI | Mistral Large, Small | ✅ | ✅ | ✅ |
| Groq | LLaMA2, Mixtral, Claude models | ❌ | ✅ | ❌ |
| Together AI | Various models | ✅ | ✅ | ❌ |
| AWS Bedrock | Claude, Titan, Llama models | ✅ | ✅ | ❌ |
| Fireworks AI | Llama, Mixtral models | ✅ | ✅ | ❌ |
| Azure OpenAI | GPT-4, GPT-3.5 models | ✅ | ✅ | ✅ |
| Vertex AI (Google) | Gemini models | ✅ | ✅ | ✅ |
| NVIDIA | Various models | ✅ | ✅ | ❌ |
| AI21 | Jamba models | ✅ | ✅ | ❌ |
| Databricks | MPT, Dolly models | ❌ | ✅ | ❌ |
| IBM watsonx.ai | Granite models | ❌ | ✅ | ❌ |
| xAI | Grok models | ✅ | ✅ | ✅ |
| Upstage | Solar models | ✅ | ✅ | ❌ |
| Hugging Face | Models with tool calling | ✅ | ✅ | ❌ |
| Ollama | Local models | ✅ | ✅ | ❌ |
| Llama.cpp | Local models | ❌ | ✅ | ❌ |

## Installation

### Option 1: Using the standalone script (Recommended)

This method creates an isolated environment without affecting your system installation:

```bash
# Clone the repository
git clone https://github.com/yourusername/browser-use-mcp.git
cd browser-use-mcp

# Run with default settings (temporary venv)
python run.py

# Run with persistent venv for faster subsequent runs
python run.py --keep-venv

# Install all providers
python run.py --all-providers

# Install a specific provider
python run.py --provider openai

# Use a specific model
python run.py --provider anthropic --model claude-3-haiku-20240307

# Use a specific .env file
python run.py --env-file /path/to/.env

# Enable debug logging
python run.py --debug
```

### Option 2: Using `pip`

Install directly with pip (with specific provider):

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with pip
pip install -e ".[openai]"  # For OpenAI
pip install -e ".[anthropic]"  # For Anthropic
pip install -e ".[all-providers]"  # All providers

# Install Playwright browsers
playwright install chromium

# Run the server
python -m browser_use_mcp
```

### Option 3: Using `pipx` for isolated installation

For an isolated global installation:

```bash
# Install using pipx
pipx install -e ".[openai]"

# Run the server
browser-use-mcp
```

## Provider Categories

Providers are organized into categories in `pyproject.toml`:

1. **Core LLM Providers**: `openai`, `anthropic`, `google`, `cohere`, `mistral`, `groq`, `together`
2. **Cloud Providers**: `aws`, `azure`, `vertex`, `fireworks`, `nvidia`, `databricks`, `ai21`, `ibm`, `xai`, `upstage`
3. **Local Models**: `huggingface`, `ollama`, `llama-cpp`
4. **Group Packages**: 
   - `local-models`: All local model providers
   - `cloud-providers`: All cloud providers
   - `all-providers`: All available providers
   - `dev`: Development tools (pytest, black, isort)

Install any provider or group:
```bash
pip install -e ".[local-models]"  # Install all local model providers
pip install -e ".[cloud-providers]"  # Install all cloud providers
pip install -e ".[dev]"  # Install development tools
```

## Selecting Models

You can specify which model to use with the `--model` or `-m` flag:

```bash
# Use GPT-4 Turbo with OpenAI
python run.py --model gpt-4-turbo

# Use Claude 3 Haiku with Anthropic
python run.py --model claude-3-haiku-20240307

# Use specific provider and model
python run.py --provider anthropic --model claude-3-haiku-20240307

# Use local Ollama model
python run.py --provider ollama --model llama3:8b
```

If no model is specified, each provider will use a default model that has been tested for browser automation.

## API Key Configuration

You need to provide at least one API key to use this server. Set the appropriate environment variable for your chosen LLM provider:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`
- Cohere: `COHERE_API_KEY`
- Mistral AI: `MISTRAL_API_KEY`
- Groq: `GROQ_API_KEY`
- Together AI: `TOGETHER_API_KEY`
- AWS Bedrock: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- Fireworks: `FIREWORKS_API_KEY`
- Azure OpenAI: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- Vertex AI: `GOOGLE_APPLICATION_CREDENTIALS`
- NVIDIA: `NVIDIA_API_KEY`
- AI21: `AI21_API_KEY`
- Databricks: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`
- IBM watsonx.ai: `WATSONX_API_KEY`
- xAI: `XAI_API_KEY`
- Upstage: `UPSTAGE_API_KEY`
- Hugging Face: `HUGGINGFACE_API_KEY`
- Ollama: Set Ollama endpoint with `OLLAMA_BASE_URL`
- Llama.cpp: Set server endpoint with `LLAMA_CPP_SERVER_URL`

You can create a `.env` file in the project directory with your API keys:

```
OPENAI_API_KEY=your_openai_key_here
# Or any other provider key
```

## Usage with Claude for Desktop

After starting the server:

1. Open Claude for Desktop
2. Go to Settings → Experimental features
3. Enable Claude API Beta
4. Enable OpenAPI schema for API
5. Set the API URL to: `http://localhost:8000/schema`

Now you can ask Claude to perform browser automation tasks like:

- "Search for the latest news about artificial intelligence"
- "Go to example.com and fill out the contact form"
- "Find and compare prices for a laptop on Amazon"

## Development

For development work, install the dev dependencies:

```bash
pip install -e ".[dev]"
```

This includes:
- Black (code formatting)
- isort (import sorting)
- pytest (testing)

Use these tools to maintain code quality:

```bash
# Format code
black browser_use_mcp/

# Sort imports
isort browser_use_mcp/

# Run tests
pytest
```

## Troubleshooting

- **API Key Issues**: Ensure your API key is correctly set in your environment variables or `.env` file.
- **Provider Not Found**: Make sure you've installed the required provider package.
- **Browser Automation Errors**: Check that Playwright is correctly installed with `playwright install chromium`.
- **Model Selection**: If you get errors about an invalid model, try using the `--model` flag to specify a valid model for your provider.
- **Debug Mode**: Use `--debug` to enable more detailed logging that can help identify issues.

## License

MIT # browser-use-mcp
