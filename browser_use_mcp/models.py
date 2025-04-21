"""
Module containing LLM provider selection and initialization logic.
"""

from langchain.chat_models.base import BaseChatModel
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_llm(model_name: Optional[str] = None) -> BaseChatModel:
    """
    Initialize and return a LangChain chat model based on available API keys.

    The function checks for various API keys in the environment and initializes
    the appropriate model if the key is found. Only one model will be initialized
    based on priority. All models in this function support both tool calling and
    structured output.

    Args:
        model_name: Optional model name to override the default model for the provider.
                   Examples: "gpt-4", "claude-3-haiku-20240307", "gemini-1.0-pro", etc.

    Returns:
        BaseChatModel: An instance of a LangChain chat model
    """
    # Check for OpenAI API key
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        use_model = model_name if model_name else "gpt-4o"
        logger.info(f"Using OpenAI {use_model}")
        return ChatOpenAI(model=use_model)

    # Check for Anthropic API key
    elif os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        use_model = model_name if model_name else "claude-3-opus-20240229"
        logger.info(f"Using Anthropic {use_model}")
        return ChatAnthropic(model=use_model)

    # Check for Google API key
    elif os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        use_model = model_name if model_name else "gemini-1.5-pro"
        logger.info(f"Using Google {use_model}")
        return ChatGoogleGenerativeAI(model=use_model)

    # Check for Cohere API key
    elif os.environ.get("COHERE_API_KEY"):
        from langchain_cohere import ChatCohere

        use_model = model_name if model_name else "command-r-plus"
        logger.info(f"Using Cohere {use_model}")
        return ChatCohere(model=use_model)

    # Check for Mistral API key
    elif os.environ.get("MISTRAL_API_KEY"):
        from langchain_mistralai import ChatMistralAI

        use_model = model_name if model_name else "mistral-large-latest"
        logger.info(f"Using Mistral {use_model}")
        return ChatMistralAI(model=use_model)

    # Check for Groq API key
    elif os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        use_model = model_name if model_name else "llama3-70b-8192"
        logger.info(f"Using Groq {use_model}")
        return ChatGroq(model=use_model)

    # Check for Together API key
    elif os.environ.get("TOGETHER_API_KEY"):
        from langchain_together import ChatTogether

        use_model = model_name if model_name else "meta-llama/Llama-3-70b-chat"
        logger.info(f"Using Together AI {use_model}")
        return ChatTogether(model=use_model)

    # Check for AWS Bedrock
    elif os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    ):
        from langchain_aws import ChatBedrock

        use_model = model_name if model_name else "anthropic.claude-3-sonnet-20240229"
        logger.info(f"Using AWS Bedrock {use_model}")
        return ChatBedrock(model_id=use_model)

    # Check for Fireworks API key
    elif os.environ.get("FIREWORKS_API_KEY"):
        from langchain_fireworks import ChatFireworks

        use_model = (
            model_name if model_name else "accounts/fireworks/models/llama-v3-70b-chat"
        )
        logger.info(f"Using Fireworks {use_model}")
        return ChatFireworks(model=use_model)

    # Check for Azure OpenAI API key
    elif os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get(
        "AZURE_OPENAI_ENDPOINT"
    ):
        from langchain_openai import AzureChatOpenAI

        model_name = (
            model_name
            if model_name
            else os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        )
        logger.info(f"Using Azure OpenAI {model_name}")
        return AzureChatOpenAI(
            azure_deployment=model_name,
            openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        )

    # Check for Vertex AI
    elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            from langchain_google_vertexai import ChatVertexAI

            use_model = model_name if model_name else "gemini-1.5-pro"
            logger.info(f"Using Google Vertex AI {use_model}")
            return ChatVertexAI(model_name=use_model)
        except Exception as e:
            logger.warning(f"Failed to initialize Google Vertex AI: {e}")

    # Check for NVIDIA AI Endpoints
    elif os.environ.get("NVIDIA_API_KEY"):
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            use_model = model_name if model_name else "meta/llama3-70b-instruct"
            logger.info(f"Using NVIDIA AI {use_model}")
            return ChatNVIDIA(model=use_model)
        except Exception as e:
            logger.warning(f"Failed to initialize NVIDIA AI: {e}")

    # Check for AI21 API key
    elif os.environ.get("AI21_API_KEY"):
        try:
            from langchain_ai21 import ChatAI21

            use_model = model_name if model_name else "j2-ultra"
            logger.info(f"Using AI21 {use_model}")
            return ChatAI21(model=use_model)
        except Exception as e:
            logger.warning(f"Failed to initialize AI21: {e}")

    # Check for Databricks
    elif os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"):
        try:
            from langchain_databricks import ChatDatabricks

            use_model = model_name if model_name else "databricks-llama-3-70b"
            logger.info(f"Using Databricks {use_model}")
            return ChatDatabricks(endpoint=use_model)
        except Exception as e:
            logger.warning(f"Failed to initialize Databricks: {e}")

    # Check for IBM watsonx.ai
    elif os.environ.get("WATSONX_API_KEY"):
        try:
            from langchain_ibm import ChatWatsonx

            use_model = model_name if model_name else "meta-llama/llama-3-70b-instruct"
            logger.info(f"Using IBM Watsonx {use_model}")
            return ChatWatsonx(model_id=use_model)
        except Exception as e:
            logger.warning(f"Failed to initialize IBM Watsonx: {e}")

    # Check for xAI
    elif os.environ.get("XAI_API_KEY"):
        try:
            from langchain_xai import ChatXAI

            use_model = model_name if model_name else "grok-1"
            logger.info(f"Using xAI {use_model}")
            return ChatXAI(model=use_model)
        except Exception as e:
            logger.warning(f"Failed to initialize xAI: {e}")

    # Check for Upstage
    elif os.environ.get("UPSTAGE_API_KEY"):
        try:
            from langchain_upstage import ChatUpstage

            use_model = model_name if model_name else "solar-1-mini-chat"
            logger.info(f"Using Upstage {use_model}")
            return ChatUpstage(model_name=use_model)
        except Exception as e:
            logger.warning(f"Failed to initialize Upstage: {e}")

    # Check for Hugging Face API key (local models supported)
    elif os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        use_model = model_name if model_name else "meta-llama/Llama-3-8b-chat-hf"
        logger.info(f"Using Hugging Face {use_model}")
        llm = HuggingFaceEndpoint(
            repo_id=use_model,
            task="text-generation",
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=llm)

    # Check for Ollama (local models)
    elif (
        os.environ.get("OLLAMA_HOST")
        or os.path.exists("/usr/local/bin/ollama")
        or os.path.exists("/usr/bin/ollama")
    ):
        try:
            from langchain_ollama import ChatOllama

            use_model = model_name if model_name else "llama3"
            logger.info(f"Using Ollama {use_model}")
            return ChatOllama(model=use_model)
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama: {e}")

    # Check for llama.cpp (local models)
    elif os.environ.get("LLAMA_CPP_MODEL_PATH"):
        try:
            from langchain_community.chat_models import ChatLlamaCpp

            model_path = (
                model_name if model_name else os.environ.get("LLAMA_CPP_MODEL_PATH")
            )
            logger.info(f"Using Llama.cpp with model at {model_path}")
            return ChatLlamaCpp(model_path=model_path)
        except Exception as e:
            logger.warning(f"Failed to initialize Llama.cpp: {e}")

    # Fallback to error if no API keys are available
    else:
        raise ValueError(
            "No API keys found. Please set one of the following environment variables:\n"
            "- OPENAI_API_KEY\n"
            "- ANTHROPIC_API_KEY\n"
            "- GOOGLE_API_KEY\n"
            "- COHERE_API_KEY\n"
            "- MISTRAL_API_KEY\n"
            "- GROQ_API_KEY\n"
            "- TOGETHER_API_KEY\n"
            "- AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
            "- FIREWORKS_API_KEY\n"
            "- AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT\n"
            "- GOOGLE_APPLICATION_CREDENTIALS (for Vertex AI)\n"
            "- NVIDIA_API_KEY\n"
            "- AI21_API_KEY\n"
            "- DATABRICKS_HOST and DATABRICKS_TOKEN\n"
            "- WATSONX_API_KEY\n"
            "- XAI_API_KEY\n"
            "- UPSTAGE_API_KEY\n"
            "- HUGGINGFACEHUB_API_TOKEN\n"
            "- OLLAMA_HOST (for local models)\n"
            "- LLAMA_CPP_MODEL_PATH (for local models)"
        )
