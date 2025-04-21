#!/usr/bin/env python3
"""
Run script for browser-use-mcp that creates an isolated environment.

This script sets up a virtual environment, installs the necessary dependencies,
and runs the browser-use-mcp server without affecting the system installation.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import venv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run browser-use-mcp in an isolated environment."
    )
    parser.add_argument(
        "--venv-path",
        type=str,
        default=None,
        help="Path to virtual environment (default: temporary directory)",
    )
    parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Keep the virtual environment after execution",
    )
    parser.add_argument(
        "--all-providers",
        action="store_true",
        help="Install all available LLM providers",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=[
            "openai",
            "anthropic",
            "google",
            "cohere",
            "mistral",
            "groq",
            "together",
            "aws",
            "fireworks",
            "azure",
            "vertex",
            "nvidia",
            "ai21",
            "databricks",
            "ibm",
            "xai",
            "upstage",
            "huggingface",
            "ollama",
            "llama-cpp",
        ],
        help="Install a specific LLM provider",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env file with API keys",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Specify the model to use with the selected provider",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def create_or_update_venv(venv_path):
    """Create or update a virtual environment."""
    venv_path = Path(venv_path)
    if venv_path.exists():
        print(f"Using existing virtual environment at {venv_path}")
    else:
        print(f"Creating virtual environment at {venv_path}")
        venv.create(venv_path, with_pip=True)
    return venv_path


def get_venv_pip(venv_path):
    """Get path to pip in the virtual environment."""
    if sys.platform == "win32":
        return str(venv_path / "Scripts" / "pip.exe")
    return str(venv_path / "bin" / "pip")


def get_venv_python(venv_path):
    """Get path to python in the virtual environment."""
    if sys.platform == "win32":
        return str(venv_path / "Scripts" / "python.exe")
    return str(venv_path / "bin" / "python")


def install_package(venv_path, provider=None, all_providers=False):
    """Install the necessary packages."""
    pip = get_venv_pip(venv_path)

    # Upgrade pip first
    subprocess.check_call([pip, "install", "--upgrade", "pip"])

    # Install package with appropriate extras
    if all_providers:
        subprocess.check_call([pip, "install", "-e", ".[all-providers]"])
    elif provider:
        subprocess.check_call([pip, "install", "-e", f".[{provider}]"])
    else:
        subprocess.check_call([pip, "install", "-e", "."])

    # Install Playwright
    python = get_venv_python(venv_path)
    subprocess.check_call([python, "-m", "playwright", "install", "chromium"])


def run_server(venv_path, env_file, model=None, debug=False):
    """Run the server using the virtual environment."""
    python = get_venv_python(venv_path)

    # Set up environment variables from .env file
    env = os.environ.copy()

    if os.path.exists(env_file):
        print(f"Loading environment variables from {env_file}")
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env[key] = value

    # Build command with arguments
    cmd = [python, "-m", "browser_use_mcp"]

    if model:
        cmd.extend(["--model", model])

    if debug:
        cmd.append("--debug")

    cmd.extend(["--env-file", env_file])

    # Run the server
    print(f"Running command: {' '.join(cmd)}")
    subprocess.call(cmd, env=env)


def main():
    """Main function to run the script."""
    args = parse_args()

    # Use a temporary directory for the virtual environment if not specified
    temp_dir = None
    if args.venv_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        venv_path = temp_dir.name
    else:
        venv_path = args.venv_path

    try:
        venv_path = create_or_update_venv(venv_path)
        install_package(venv_path, args.provider, args.all_providers)
        run_server(venv_path, args.env_file, args.model, args.debug)
    finally:
        if temp_dir and not args.keep_venv:
            print("Cleaning up temporary virtual environment")
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
