import os
from deepseek_llm import DeepSeekLLM

def get_llm(model_name="claude-3.5-sonnet"):
    """Returns a configured LLM instance based on model selection"""
    
    # Model configurations for ablation study
    model_configs = {
        "claude-3.5-sonnet": {
            "provider": "openrouter",
            "model": "anthropic/claude-3.5-sonnet",
            "temperature": 0
        },
        "gpt-4-turbo": {
            "provider": "openrouter", 
            "model": "openai/gpt-4-turbo-preview",
            "temperature": 0
        },
        "llama-3.1-70b": {
            "provider": "openrouter",
            "model": "meta-llama/llama-3.1-70b-instruct",
            "temperature": 0  
        },
        "deepseek-r1": {
            "provider": "openrouter",
            "model": "deepseek/deepseek-r1",
            "temperature": 0,
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(model_configs.keys())}")
    
    config = model_configs[model_name]
    
    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
    if not openrouter_api_key:
        raise ValueError("No OpenRouter API key found. Make sure OPENROUTER_API_KEY is in your .env file.")
    
    return DeepSeekLLM(
        provider=config["provider"],
        model=config["model"],
        temperature=config["temperature"],
        api_key=openrouter_api_key
    )