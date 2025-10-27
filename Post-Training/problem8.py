from problem2 import BPETokenizer
from collections import Counter
import math
import numpy as np

corpus = [
    # Original
    "The quick brown fox jumps over the lazy dog.",
    
    # Exact duplicate
    "The quick brown fox jumps over the lazy dog",
    
    # Duplicate with slight variation
    "The quick brown fox jumps over the lazy dog!",
    
    # Different documents
    "To be or not to be, that is the question.",

    "The early bird catches a worm.",

    "To be or not to be, the early bird catches many worm.",

    # Another exact duplicate
    "To be or not to be, that is the question.",
    
    # Near duplicate (small change)
    "To be or not to be, that is the question!",
    
    # Similar content but different enough
    "Something completely different here.",
    
    # Another duplicate
    "The early bird catches the worm.",
    
    # Slightly modified
    "The early bird catches many worm.",

    "The quick question cannot catch the right question."

    "The quick question catches a human in the worm."
]

if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(corpus)

    tfs = []
    freq_mapping = Counter()
    for doc in corpus:
        token_ids = tokenizer.encode(doc)
        tf = Counter(token_ids)
        tfs.append(tf)
        freq_mapping.update(tf)

    freq_mapping = {token: math.log(len(corpus) / count) for token, count in freq_mapping.items()}

    arrays = []
    for tf in tfs:
        array = np.zeros(len(tokenizer.vocab))
        for token, count in tf.items():
            array[token] = count * freq_mapping[token]
        arrays.append(array)
    arrays = np.stack(arrays, 0)

    query_text = "The quick brown question catches the worm."
    query = tokenizer.encode(query_text)
    counter = Counter(query)
    query = np.zeros(len(tokenizer.vocab))
    for token, count in counter.items():
        # Option 1: Query without IDF (current approach, most common)
        query[token] = count
        
        # Option 2: Query with IDF (uncomment to try):
        # if token in freq_mapping:
        #     query[token] = count * freq_mapping[token]

    scores = query @ arrays.T

    print(query_text)
    top_k = np.argsort(scores)[::-1][:5]
    for idx in top_k:
        print(f"Score: {scores[idx]}, Document: {corpus[idx]}")