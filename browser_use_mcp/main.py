from mcp.server.fastmcp import FastMCP
from browser_use import Agent
import logging
import argparse

# Import the get_llm function from models.py
from browser_use_mcp.models import get_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("browser-use")

# Global variable for the LLM instance
llm = None

@mcp.tool()
async def run_browser_use_task(task: str) -> str:
    """Run a browser automation task described in natural language using browser_use Agent, it can perform many tasks like browsing the web, searching the web, and more.

    Args:
        task: Natural language description of the task to perform
    """
    try:
        # Log that we're starting the task
        logger.info(f"Starting browser task: {task}")
        # Create and run the browser_use agent
        agent = Agent(task=task, llm=llm)
        result = await agent.run()
        return f"Task completed: {result}"
    except Exception as e:
        logger.error(f"Error in async browser task: {str(e)}")
        return f"Error performing task: {str(e)}"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Browser Use MCP Server")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Specify the model to use. Overrides the default model for the provider.",
    )

    return parser.parse_args()


def main():
    """Entry point for the command-line tool"""
    args = parse_args()


    # Initialize the LLM with the specified model
    global llm
    try:
        llm = get_llm(model_name=args.model)
    except ValueError as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        return 1

    logger.info("Starting browser-use-mcp server")
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
