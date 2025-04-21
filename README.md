<div align="center">
  <br />
  <br />
  <img src="public/light.svg" alt="Browser Use MCP Server" width="100%">
  <br />
  <br />
</div>

# Browser Use MCP Server

A FastMCP server that enables browser automation through natural language commands. This server allows Language Models to browse the web, fill out forms, click buttons, and perform other web-based tasks via a simple API.

## Quick Start

### 1. Install the package

Install with a specific provider (e.g., OpenAI)

```bash
pip install -e "git+https://github.com/yourusername/browser-use-mcp.git#egg=browser-use-mcp[openai]"
```
Or install all providers
```bash

pip install -e "git+https://github.com/yourusername/browser-use-mcp.git#egg=browser-use-mcp[all-providers]"
```
Install Playwright browsers
```bash
playwright install chromium
```

### 2. Configure your MCP client

Add the browser-use-mcp server to your MCP client configuration:

```javascript
{
    "mcpServers": {
        "browser-use-mcp": {
            "command": "browser-use-mcp",
            "args": ["--model", "gpt-4o"],
            "env": {
                "OPENAI_API_KEY": "your-openai-api-key",  // Or any other provider's API key
                "DISPLAY": ":0"  // For GUI environments
            }
        }
    }
}
```

Replace `"your-openai-api-key"` with your actual API key or use an environment variable reference like `process.env.OPENAI_API_KEY`.

### 3. Use it with your favorite MCP client

#### Example using mcp-use with Python

```python
import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

async def main():
    # Load environment variables
    load_dotenv()

    # Create MCPClient from config file
    client = MCPClient(
        config={
            "mcpServers": {
                "browser-use-mcp": {
                    "command": "browser-use-mcp",
                    "args": ["--model", "gpt-4o"],
                    "env": {
                        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                        "DISPLAY": ":0",
                    },
                }
            }
        }
    )

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Run the query
    result = await agent.run(
        """
        Navigate to https://github.com, search for "browser-use-mcp", and summarize the project.
        """,
        max_steps=30,
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Using Claude for Desktop

1. Open Claude for Desktop
2. Go to Settings → Experimental features
3. Enable Claude API Beta and OpenAPI schema for API
4. Add the following configuration to your Claude Desktop config file:
   - Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%AppData%\Claude\claude_desktop_config.json`

```json
{
    "mcpServers": {
        "browser-use": {
            "command": "browser-use-mcp",
            "args": ["--model", "claude-3-opus-20240229"]
        }
    }
}
```

5. Start a new conversation with Claude and ask it to perform web tasks

## Supported LLM Providers

The following LLM providers are supported for browser automation:

| Provider | API Key Environment Variable |
|----------|----------------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| Mistral AI | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` |
| Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS` |
| NVIDIA | `NVIDIA_API_KEY` |
| AI21 | `AI21_API_KEY` |
| Databricks | `DATABRICKS_HOST` and `DATABRICKS_TOKEN` |
| IBM watsonx.ai | `WATSONX_API_KEY` |
| xAI | `XAI_API_KEY` |
| Upstage | `UPSTAGE_API_KEY` |
| Hugging Face | `HUGGINGFACE_API_KEY` |
| Ollama | `OLLAMA_BASE_URL` |
| Llama.cpp | `LLAMA_CPP_SERVER_URL` |

For more information check out: https://python.langchain.com/docs/integrations/chat/

You can create a `.env` file in the project directory with your API keys:

```
OPENAI_API_KEY=your_openai_key_here
# Or any other provider key
```

## Troubleshooting

- **API Key Issues**: Ensure your API key is correctly set in your environment variables or `.env` file.
- **Provider Not Found**: Make sure you've installed the required provider package.
- **Browser Automation Errors**: Check that Playwright is correctly installed with `playwright install chromium`.
- **Model Selection**: If you get errors about an invalid model, try using the `--model` flag to specify a valid model for your provider.
- **Debug Mode**: Use `--debug` to enable more detailed logging that can help identify issues.
- **MCP Client Configuration**: Make sure your MCP client is correctly configured with the right command and environment variables.

## License

MIT # browser-use-mcp
