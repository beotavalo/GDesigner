import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')

# Debug: Print environment variable values
print(f"Environment check - BASE_URL: {MINE_BASE_URL}")
print(f"Environment check - API_KEY: {'SET' if MINE_API_KEYS else 'NOT SET'}")


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    
    # Check if environment variables are properly set
    if not request_url or not authorization_key:
        raise ValueError("BASE_URL and API_KEY environment variables must be set. Please create a .env file with these variables.")
    
    print(f"Making request to: {request_url}")
    print(f"Using authorization key: {authorization_key[:10]}..." if authorization_key else "No authorization key")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {authorization_key}'
    }
    data = {
        "model": model,
        "messages": msg,
        "stream": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(request_url, headers=headers ,json=data) as response:
                print(f"Response status: {response.status}")
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                response_data = await response.json()
                prompt = "".join([item['content'] for item in msg])
                # Extract the response text from OpenAI API format
                response_text = response_data['choices'][0]['message']['content']
                cost_count(prompt, response_text, model)
                return response_text
    except Exception as e:
        print(f"Exception in achat: {type(e).__name__}: {str(e)}")
        raise

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        
        # Convert messages to dict format expected by achat
        msg_dicts = []
        for msg in messages:
            if isinstance(msg, dict):
                msg_dicts.append(msg)
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                # It's a Message object
                msg_dicts.append({"role": msg.role, "content": msg.content})
            else:
                raise ValueError(f"Invalid message format: {type(msg)}")
        
        return await achat(self.model_name, msg_dicts)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        # Convert messages to dict format expected by achat
        msg_dicts = []
        for msg in messages:
            if isinstance(msg, dict):
                msg_dicts.append(msg)
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                # It's a Message object
                msg_dicts.append({"role": msg.role, "content": msg.content})
            else:
                raise ValueError(f"Invalid message format: {type(msg)}")
        
        # Note: This is a synchronous wrapper around the async achat function
        # In a real implementation, you might want to use asyncio.run() or similar
        raise NotImplementedError("Synchronous gen method not implemented. Use agen() instead.")