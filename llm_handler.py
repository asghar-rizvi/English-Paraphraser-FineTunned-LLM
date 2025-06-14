import random
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import torch

class LLM_Handler:
    def __init__(self, peft_model_id: str, device: Optional[str] = None):
        
        self.peft_model_id = peft_model_id
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        try:
            config = PeftConfig.from_pretrained(self.peft_model_id)
        
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                config.base_model_name_or_path
            ).to(self.device)
            
            self.model = PeftModel.from_pretrained(
                base_model, 
                self.peft_model_id
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.peft_model_id)
            
            self.model = self.model.merge_and_unload()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def paraphrase(
        self,
        text: str,
        num_return_sequences: int = 3,
        temperature: float = 0.9,
        max_length: int = 128,
        prompt_style: Optional[str] = None
    ) -> List[str]:
        
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        prompts = {
            "technical": "Paraphrase this technically: ",
            "formal": "Rewrite this formally: ",
            "simple": "Simplify this text: ",
            "creative": "Rephrase this creatively: ",
            "neutral": "Express this differently: "
        }
        
        if prompt_style and prompt_style in prompts:
            prompt = prompts[prompt_style]
        else:
            prompt = random.choice(list(prompts.values()))
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt + text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length"
            ).to(self.device)

            # Generate outputs
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=num_return_sequences,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            # Decode and clean outputs
            paraphrases = []
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                
                # Remove any leftover prompt
                for p in prompts.values():
                    decoded = decoded.replace(p, "")
                paraphrases.append(decoded.strip())

            return paraphrases

        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")

    def to(self, device: str):
        if self.model:
            self.device = device
            self.model = self.model.to(device)
        return self
    
if __name__ == '__main__' :
    handler = LLM_Handler('model/final_paraphraser')
    text = """The rapid development of artificial intelligence has raised significant ethical concerns,
    particularly regarding data privacy and algorithmic bias. Many experts argue that comprehensive
    regulations need to be established before these technologies become more pervasive in society."""
    
    paraphrases = handler.paraphrase(text)  
    
    print("Original:\n", text)
    print("\nParaphrases:")
    for i, para in enumerate(paraphrases, 1):
        print(f"\n{i}. {para}")
    