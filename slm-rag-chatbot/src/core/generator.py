from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
import sys
sys.path.insert(0, 'C:\\Users\\avina\\Desktop\\AI Projects\\slm-rag-chatbot')
from config.settings import settings

class ResponseGenerator:
    """Generate responses using Llama 3.2 3B"""
    
    def __init__(self, model_path: str = "G:/llama-models/llama-3.2-3b"):
        print(f"Loading Llama 3.2 3B from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # CPU-optimized loading
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # CPU works better with float32
            low_cpu_mem_usage=True
        )
        
        print("✓ Llama 3.2 3B loaded (8-bit)")
    
    def generate(
        self,
        query: str,
        context: List[Dict],
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate response given query and context"""
        
        # Build context from top 3 results
        context_str = ""
        for i, doc in enumerate(context[:3]):
            context_str += f"\n[Source {i+1}]:\n{doc['text']}\n"
        
        # Llama 3.2 chat template
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Answer based on the provided context. Be concise and cite sources.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context_str}

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "assistant<|end_header_id|>" in full_response:
            answer = full_response.split("assistant<|end_header_id|>")[-1].strip()
        else:
            answer = full_response[len(prompt):].strip()
        
        return answer
