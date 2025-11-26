"""
DeepSeek API Client
"""

import os
import requests
import json
import time
from typing import List, Dict, Optional


class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")
        
        self.api_base = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def send_request(
        self, 
        context: List[Dict[str, str]], 
        message: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        max_retries: int = 3
    ) -> Optional[str]:
        context.append({"role": "user", "content": message})
        
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": context,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=self.headers, timeout=180)
                response.raise_for_status()
                
                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
                context.append({"role": "assistant", "content": content})
                
                print(f"[DeepSeek] {content[:200]}...")
                if 'usage' in response_data:
                    usage = response_data['usage']
                    print(f"[Tokens] {usage.get('total_tokens', 0)}")
                
                return content
                
            except Exception as e:
                print(f"[ERROR] Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        
        return None


_client = None

def get_client() -> DeepSeekClient:
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client


def send_request(
    context: List[Dict[str, str]], 
    message: str,
    temperature: float = 0.7,
    max_tokens: int = 4000
) -> Optional[str]:
    client = get_client()
    return client.send_request(context, message, temperature, max_tokens)


if __name__ == "__main__":
    if not os.getenv('DEEPSEEK_API_KEY'):
        print("❌ Set DEEPSEEK_API_KEY first")
        exit(1)
    
    context = []
    response = send_request(context, "Say 'Hello!'", temperature=0.3)
    print(f"✅ Test {'passed' if response else 'failed'}")