import requests
from typing import List, Dict

class ModalGenerator:
    """Generator that calls Modal GPU endpoint"""
    
    def __init__(self):
        self.endpoint = "https://avipalsai--llm-rag-generator-generate.modal.run"
        print("✓ Modal GPU generator ready")
    
    def generate(
        self,
        query: str,
        context: List[Dict],
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate response using Modal GPU"""
        
        if not context:
            return "I don't have enough information to answer that question."
        
        # Combine context
        context_text = "\n\n".join([c['text'] for c in context])
        
        # Call Modal
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "query": query,
                    "context": context_text,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["answer"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling Modal: {str(e)}"
