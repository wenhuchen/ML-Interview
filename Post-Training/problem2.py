"""
Simple BPE (Byte Pair Encoding) Tokenizer Implementation from Scratch

This implementation demonstrates how to learn a tokenizer for an LLM without using
any external libraries like transformers, huggingface, or PyTorch.
"""

import re
from collections import Counter, defaultdict
import token
from typing import Any, List, Dict, Tuple, final


class BPETokenizer:
    """Simple BPE tokenizer implementation from scratch"""
    
    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.merges = []  # List of merge operations
        self.merges_lookup = {}  # Map from pair to merge priority (lower = earlier merge)
        self.vocab = {}  # Map from token to index
        self.inv_vocab = {}
        
    def get_word_freqs(self, corpus: List[str]) -> Dict[str, int]:
        """Count word frequencies in the corpus"""
        word_freqs = {}
        for text in corpus:
            # Split by whitespace and punctuation
            words = re.findall(r'\S+', text.lower())
            for word in words:
                word_freqs[word] = word_freqs.get(word, 0) + 1
        return word_freqs
    
    def get_splits(self, word: str) -> List[str]:
        """Split a word into characters"""
        return [c for c in word] + ['</w>']  # End of word marker
    
    def get_stats(self, splits_dict: Dict) -> Dict:
        """Get statistics of adjacent pairs"""
        pairs = defaultdict(int)
        for word, splits in splits_dict.items():
            for i in range(len(splits) - 1):
                pairs[(splits[i], splits[i+1])] += self.word_freqs.get(word, 1)
        return pairs
    
    def merge_pair(self, splits_dict: Dict, pair: Tuple[str, str]) -> Dict:
        """Merge a pair of adjacent units"""
        new_splits = {}
        bigram = pair
        for word, splits in splits_dict.items():
            new_splits_list = []
            i = 0
            while i < len(splits):
                if i < len(splits) - 1 and splits[i] == bigram[0] and splits[i+1] == bigram[1]:
                    new_splits_list.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    new_splits_list.append(splits[i])
                    i += 1
            new_splits[word] = new_splits_list
        return new_splits
    
    def train(self, corpus: List[str]):
        """Train the BPE tokenizer on the corpus"""
        print("Training BPE tokenizer...")
        
        # Step 1: Get word frequencies
        self.word_freqs = self.get_word_freqs(corpus)
        print(f"Number of unique words: {len(self.word_freqs)}")
        
        # Step 2: Initialize splits
        splits_dict = {}
        for word in self.word_freqs.keys():
            splits_dict[word] = self.get_splits(word)
        
        # Step 3: Calculate how many merges are needed
        # Estimate: we start with characters and will need many merges to reach vocab_size
        initial_vocab_size = len(set[Any](token for splits in splits_dict.values() for token in splits))
        merges_needed = max(0, self.vocab_size - initial_vocab_size)
        print(f"Number of merges needed: {merges_needed}")
 
        # Step 4: Perform BPE merges
        for i in range(merges_needed):
            # Get statistics
            pairs = self.get_stats(splits_dict)
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Record the merge
            self.merges.append(best_pair)
            
            # Update splits (this is what actually matters for the vocabulary)
            splits_dict = self.merge_pair(splits_dict, best_pair)
            
            if (i + 1) % 50 == 0:
                print(f"  Completed {i + 1} merges...")

        # Create final vocabulary
        final_vocab = set()
        for word, splits in splits_dict.items():
            final_vocab.update(splits)
        
        # Add all ASCII letters to handle any OOV words (BPE can decompose to characters)
        # This ensures any word can be tokenized down to characters
        for char in 'abcdefghijklmnopqrstuvwxyz':
            final_vocab.add(char)
            final_vocab.add(char + '</w>')
        
        # Add end-of-word marker if not already present
        final_vocab.add('</w>')
        
        # Create vocabulary mapping
        self.vocab = {token: idx for idx, token in enumerate(sorted(final_vocab))}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        # Create lookup table for O(1) pair lookup
        self.merges_lookup = {pair: idx for idx, pair in enumerate(self.merges)}

        print(f"Training complete! Final vocabulary size: {len(self.vocab)}")
    
    def apply_bpe(self, word: str) -> List[str]:
        """Apply BPE encoding to a single word"""
        # Start with character-level splits
        splits = self.get_splits(word)
        
        if not self.merges_lookup:
            return splits
        
        # Apply merges until no more merges can be found
        while True:
            # Find the highest priority merge available in current splits
            best_pair = None
            best_priority = len(self.merges)
            
            for i in range(len(splits) - 1):
                pair = (splits[i], splits[i+1])
                merge_priority = self.merges_lookup.get(pair)
                if merge_priority is not None and merge_priority < best_priority:
                    best_pair = pair
                    best_priority = merge_priority
                # Note: We continue scanning to ensure we pick the highest 
                # priority merge (lowest index) among all available pairs
            
            if best_pair is None:
                break
                
            # Apply the merge at all positions where it appears
            new_splits = []
            i = 0
            while i < len(splits):
                if i < len(splits) - 1 and (splits[i], splits[i+1]) == best_pair:
                    new_splits.append(splits[i] + splits[i+1])
                    i += 2
                else:
                    new_splits.append(splits[i])
                    i += 1
            
            splits = new_splits
        
        return splits
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs"""
        words = re.findall(r'\S+', text.lower())
        token_ids = []
        
        for word in words:
            splits = self.apply_bpe(word)
            for token in splits:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    spacing = False
                    if '</w>' in token:
                        token = token.replace('</w>', '')
                        spacing = True
                    for char in token:
                        token_ids.append(self.vocab[char])
                    if spacing:
                        token_ids.append(self.vocab['</w>'])
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        # Reverse vocabulary lookup
        token_ids = [idx for idx in token_ids if idx != self.vocab['<pad>']]
        tokens = [self.inv_vocab[idx] for idx in token_ids]
        # Join tokens and remove end-of-word markers
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()

    def expand_vocab(self, new_tokens: List[str]):
        """Expand the vocabulary with new tokens"""
        for token in new_tokens:
            self.vocab[token] = len(self.vocab)
            self.inv_vocab[len(self.vocab) - 1] = token
            self.vocab_size += 1
        
        print('Expanded vocabulary to size ', len(self.vocab))


def example_usage():
    """Demonstrate the tokenizer"""
    
    # Sample corpus for training
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "a brown cat jumps over a lazy cat",
        "the dog is quick and brown",
        "jumps jumps jumps over over over",
        "hello world from the tokenizer",
        "the cat and the dog are friends",
        "quick brown fox are quick",
        "tokenization is important for NLP",
        "NLP uses tokenization for text processing",
        "machine learning models need tokenization",
    ]
    
    print("=" * 60)
    print("BPE Tokenizer Training Example")
    print("=" * 60)
    
    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(corpus)
    
    print("\n" + "=" * 60)
    print("Testing Tokenization")
    print("=" * 60)
    
    # Test encoding
    test_text = "the quick brown fox jumps"
    token_ids = tokenizer.encode(test_text)
    print(f"\nOriginal: '{test_text}'")
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    
    # Test decoding
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: '{decoded}'")
    
    print("\n" + "=" * 60)
    print("Sample Vocabulary")
    print("=" * 60)
    
    # Show some vocabulary items
    vocab_list = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
    print("\nSample tokens from vocabulary:")
    for i, (token, idx) in enumerate(vocab_list[:30]):
        print(f"  {idx:3d}: '{token}'")
    print(f"  ... (showing first 30 of {len(vocab_list)} tokens)")
    
    print("\n" + "=" * 60)
    print("Some BPE Merges Learned")
    print("=" * 60)
    for i, merge in enumerate(tokenizer.merges[:20]):
        print(f"  Merge {i+1}: '{merge[0]}' + '{merge[1]}' -> '{merge[0] + merge[1]}'")
    print(f"  ... (showing first 20 of {len(tokenizer.merges)} merges)")
    
    # Demonstrate encoding of novel words
    print("\n" + "=" * 60)
    print("Handling Novel Words")
    print("=" * 60)
    
    novel_texts = [
        "the quick dog",
        "brown cat jumps",
        "novel word tokenization",
        "How is the weather today"
    ]
    
    for text in novel_texts:
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        print(f"Text: '{text}'")
        print(f"  Token IDs: {token_ids}")
        print(f"  Decoded: '{decoded}'")
        print()


if __name__ == "__main__":
    example_usage()

