import os
import json
import sentencepiece as spm
from typing import List, Optional


class JapaneseTokenizer:
    def __init__(self, model_path: Optional[str] = None):
        self.sp = None
        self.model_path = model_path
        self.vocab_size = 32000
        self.special_tokens = {
            "<|endoftext|>": 0,
            "<|pad|>": 1,
            "<|unk|>": 2,
            "<|bos|>": 3,
            "<|eos|>": 4,
        }
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(self, texts: List[str], model_prefix: str = "japanese_bpe", vocab_size: int = 32000):
        """
        Train a SentencePiece BPE model on Japanese texts.
        
        Args:
            texts: List of training texts
            model_prefix: Prefix for saved model files
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        
        # Write texts to temporary file
        temp_file = "temp_train_data.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text.strip() + "\n")
        
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=self.special_tokens["<|pad|>"],
            unk_id=self.special_tokens["<|unk|>"],
            bos_id=self.special_tokens["<|bos|>"],
            eos_id=self.special_tokens["<|eos|>"],
            user_defined_symbols=["<|endoftext|>"]
        )
        
        # Load the trained model
        self.model_path = f"{model_prefix}.model"
        self.load(self.model_path)
        
        # Clean up temporary file
        os.remove(temp_file)
        
    def load(self, model_path: str):
        """Load a trained SentencePiece model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.model_path = model_path
        
    def save(self, path: str):
        """Save tokenizer configuration."""
        config = {
            "model_path": self.model_path,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_config(cls, config_path: str):
        """Load tokenizer from configuration file."""
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        tokenizer = cls(config["model_path"])
        tokenizer.vocab_size = config["vocab_size"]
        tokenizer.special_tokens = config["special_tokens"]
        return tokenizer
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if self.sp is None:
            raise ValueError("Tokenizer model not loaded. Train or load a model first.")
        
        # Encode with SentencePiece
        tokens = self.sp.encode_as_ids(text)
        
        # Add special tokens if requested
        if add_special_tokens:
            tokens = [self.special_tokens["<|bos|>"]] + tokens + [self.special_tokens["<|eos|>"]]
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if self.sp is None:
            raise ValueError("Tokenizer model not loaded. Train or load a model first.")
        
        # Filter special tokens if requested
        if skip_special_tokens:
            special_ids = set(self.special_tokens.values())
            token_ids = [tid for tid in token_ids if tid not in special_ids]
        
        # Decode with SentencePiece
        text = self.sp.decode_ids(token_ids)
        return text
    
    def batch_encode(self, texts: List[str], max_length: int = 1024, 
                    padding: bool = True, truncation: bool = True) -> dict:
        """
        Batch encode multiple texts with padding and truncation.
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            padding: Whether to pad sequences to max_length
            truncation: Whether to truncate sequences longer than max_length
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        encoded_texts = []
        attention_masks = []
        
        for text in texts:
            tokens = self.encode(text, add_special_tokens=True)
            
            # Truncate if necessary
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(tokens)
            
            # Pad if necessary
            if padding and len(tokens) < max_length:
                padding_length = max_length - len(tokens)
                tokens.extend([self.special_tokens["<|pad|>"]] * padding_length)
                attention_mask.extend([0] * padding_length)
            
            encoded_texts.append(tokens)
            attention_masks.append(attention_mask)
        
        return {
            "input_ids": encoded_texts,
            "attention_mask": attention_masks
        }
    
    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<|pad|>"]
    
    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<|eos|>"]
    
    @property
    def bos_token_id(self) -> int:
        return self.special_tokens["<|bos|>"]
    
    @property
    def unk_token_id(self) -> int:
        return self.special_tokens["<|unk|>"]
    
    def __len__(self) -> int:
        return self.vocab_size