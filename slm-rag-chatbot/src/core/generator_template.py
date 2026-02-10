from typing import List, Dict

class TemplateGenerator:
    """Template-based responses (for CPU testing, replace with Llama on GPU)"""
    
    def __init__(self):
        print("✓ Template generator ready (GPU Llama deployment tomorrow)")
    
    def generate(
        self,
        query: str,
        context: List[Dict],
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate response using retrieved context"""
        
        if not context:
            return "I don't have enough information to answer that question."
        
        # Use top result
        top_result = context[0]['text']
        
        # Simple template response
        answer = f"Based on the information provided: {top_result}"
        
        if len(context) > 1:
            answer += f"\n\nAdditional context: {context[1]['text'][:100]}..."
        
        return answer
