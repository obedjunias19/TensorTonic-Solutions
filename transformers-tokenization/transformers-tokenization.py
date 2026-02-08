import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        vocab_list = []
        for sentence in texts:
            vocab_list.extend(sentence.split())
        
        
        vocab = set(sorted(vocab_list))

        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        
        n = 4
        i = 0
        for token in vocab: 
            self.word_to_id[token] = n + i
            i+=1

        for w, i in self.word_to_id.items():
            self.id_to_word[i] = w
        self.vocab_size = len(self.word_to_id)


    
    def encode(self, text: str) -> List[int]:

        return [
            self.word_to_id[tokens] if tokens in self.word_to_id else self.word_to_id[self.unk_token] for tokens in text.split()
        ]
    
    def decode(self, ids: List[int]) -> str:

        result = [self.id_to_word[id_] for id_ in ids]

        return " ".join(result)
        
