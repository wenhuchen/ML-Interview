"""
MinHash utilities for near-duplicate detection.

Provides an LSH-style MinHash implementation to estimate Jaccard
similarity between document shingles for deduplication.
"""

import hashlib
import re
from typing import List, Set, Tuple
from collections import defaultdict


class MinHash:
    """
    MinHash implementation for document deduplication.
    MinHash is an LSH technique that estimates Jaccard similarity between sets.
    """
    
    def __init__(self, num_perm: int = 128):
        """
        Initialize MinHash with num_perm hash functions.
        
        Args:
            num_perm: Number of permutation functions to use (more = more accurate)
        """
        self.num_perm = num_perm
        # Generate hash functions using different seeds
        # We'll use the common MinHash approach with linear hash functions
        self.hash_funcs = self._generate_hash_functions(num_perm)
    
    def _generate_hash_functions(self, n: int) -> List[Tuple[int, int]]:
        """
        Generate n hash functions of the form h(x) = (a * x + b) mod prime
        Returns list of (a, b) pairs.
        """
        # Use a large prime number
        prime = 2147483647  # 2^31 - 1
        
        hash_funcs = []
        for i in range(n):
            # Use SHA256 to generate deterministic seeds
            seed = hashlib.sha256(f"minhash_{i}".encode()).digest()
            a = int.from_bytes(seed[:4], 'big') % prime
            b = int.from_bytes(seed[4:8], 'big') % prime
            if a == 0:
                a = 1  # Ensure a != 0
            hash_funcs.append((a, b))
        
        return hash_funcs
    
    def _compute_hash(self, value: str, a: int, b: int) -> int:
        """Compute hash value h(x) = (a * x + b) mod prime"""
        prime = 2147483647
        x = int(hashlib.md5(value.encode()).hexdigest(), 16) % prime
        return (a * x + b) % prime
    
    def compute_signature(self, document: Set[str]) -> List[int]:
        """
        Compute MinHash signature for a document (set of shingles).
        
        Args:
            document: Set of shingles (e.g., words, n-grams)
        
        Returns:
            MinHash signature vector of length num_perm
        """
        signature = [float('inf')] * self.num_perm
        
        for shingle in document:
            for i, (a, b) in enumerate(self.hash_funcs):
                hash_val = self._compute_hash(shingle, a, b)
                signature[i] = min(signature[i], hash_val)
        
        return signature
    
    def similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """
        Estimate Jaccard similarity using MinHash signatures.
        
        Args:
            sig1, sig2: MinHash signatures
        
        Returns:
            Estimated Jaccard similarity in [0, 1]
        """
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have the same length")
        
        matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return matches / len(sig1)


class DocumentProcessor:
    """Processes documents for deduplication."""
    
    @staticmethod
    def create_shingles(text: str, k: int = 3) -> Set[str]:
        """
        Convert text to k-shingles (character-level n-grams).
        
        Args:
            text: Input text
            k: Size of shingles
        
        Returns:
            Set of k-shingles
        """
        # Normalize text: lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        if len(text) < k:
            return {text}
        
        shingles = set()
        for i in range(len(text) - k + 1):
            shingles.add(text[i:i+k])
        
        return shingles
    
    @staticmethod
    def create_word_shingles(words: List[str], k: int = 2) -> Set[str]:
        """
        Create word-level k-shingles.
        
        Args:
            words: List of words
            k: Size of shingles
        
        Returns:
            Set of k-shingles as strings
        """
        if len(words) < k:
            return {" ".join(words)}
        
        shingles = set()
        for i in range(len(words) - k + 1):
            shingles.add(" ".join(words[i:i+k]))
        
        return shingles


class Deduplicator:
    """Deduplicate documents using MinHash."""
    
    def __init__(self, num_perm: int = 128, similarity_threshold: float = 0.9):
        """
        Initialize deduplicator.
        
        Args:
            num_perm: Number of hash functions for MinHash
            similarity_threshold: Documents with similarity >= threshold are considered duplicates
        """
        self.minhash = MinHash(num_perm)
        self.similarity_threshold = similarity_threshold
        self.processor = DocumentProcessor()
    
    def deduplicate(self, documents: List[str], use_word_shingles: bool = False) -> Tuple[List[str], List[int]]:
        """
        Deduplicate a corpus of documents.
        
        Args:
            documents: List of document strings
            use_word_shingles: If True, use word-level shingles; else use character-level
        
        Returns:
            Tuple of (unique_documents, original_indices)
        """
        print(f"Processing {len(documents)} documents...")
        
        # Step 1: Convert documents to shingles and compute signatures
        signatures = []
        for doc in documents:
            if use_word_shingles:
                words = re.findall(r'\w+', doc)
                shingles = self.processor.create_word_shingles(words, k=2)
            else:
                shingles = self.processor.create_shingles(doc, k=5)
            
            sig = self.minhash.compute_signature(shingles)
            signatures.append(sig)
        
        print(f"Computed MinHash signatures.")
        
        # Step 2: Cluster similar documents
        # Group documents by similar signatures
        clusters = self._cluster_by_similarity(signatures)
        
        print(f"Found {len(clusters)} unique clusters.")
        
        # Step 3: Keep one document per cluster (the first one)
        unique_docs = []
        original_indices = []
        
        for cluster in clusters:
            unique_docs.append(documents[cluster[0]])
            original_indices.append(cluster[0])
        
        return unique_docs, original_indices
    
    def _cluster_by_similarity(self, signatures: List[List[int]]) -> List[List[int]]:
        """
        Cluster documents by MinHash similarity using Union-Find.
        
        This ensures transitive similarity: if A~B and B~C, then A, B, C 
        will be in the same cluster even if A~C < threshold.
        
        Args:
            signatures: List of MinHash signatures
        
        Returns:
            List of clusters, where each cluster is a list of document indices
        """
        n = len(signatures)
        
        # Union-Find implementation
        parent = list(range(n))
        
        def find(x):
            """Find root with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            """Union two sets."""
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y
        
        # Build similarity graph and connect similar documents
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.minhash.similarity(signatures[i], signatures[j])
                if sim >= self.similarity_threshold:
                    union(i, j)
        
        # Group documents by their roots
        clusters_dict = defaultdict(list)
        for i in range(n):
            root = find(i)
            clusters_dict[root].append(i)
        
        # Convert to list of lists, sorted by smallest index in each cluster
        clusters = sorted(clusters_dict.values(), key=lambda x: min(x))
        
        return clusters
    
    def find_duplicates(self, documents: List[str]) -> List[Tuple[int, int, float]]:
        """
        Find all duplicate pairs and their similarities.
        
        Args:
            documents: List of document strings
        
        Returns:
            List of (idx1, idx2, similarity) tuples
        """
        # Compute signatures
        signatures = []
        for doc in documents:
            shingles = self.processor.create_shingles(doc, k=5)
            sig = self.minhash.compute_signature(shingles)
            signatures.append(sig)
        
        # Find duplicates
        duplicates = []
        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                sim = self.minhash.similarity(signatures[i], signatures[j])
                if sim >= self.similarity_threshold:
                    duplicates.append((i, j, sim))
        
        return duplicates


def demo():
    """Demonstrate MinHash for corpus deduplication."""
    
    # Create a corpus with some duplicate and near-duplicate documents
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
    ]
    
    print("=" * 80)
    print("MinHash Deduplication Demo")
    print("=" * 80)
    
    print("\nOriginal corpus:")
    for i, doc in enumerate(corpus):
        print(f"{i}: {doc}")
    
    print("\n" + "=" * 80)
    print("Running deduplication...")
    print("=" * 80)
    
    # Initialize deduplicator
    deduplicator = Deduplicator(
        num_perm=256,  # More permutations = more accurate
        similarity_threshold=0.8  # 80% similarity threshold
    )
    
    # Deduplicate
    unique_docs, indices = deduplicator.deduplicate(corpus)
    
    print(f"\nOriginal documents: {len(corpus)}")
    print(f"Unique documents: {len(unique_docs)}")
    print(f"Removed: {len(corpus) - len(unique_docs)} duplicates")
    
    print("\nUnique documents:")
    for i, (doc, orig_idx) in enumerate(zip(unique_docs, indices)):
        print(f"{i}: [{orig_idx}] {doc}")
    
    print("\n" + "=" * 80)
    print("Finding duplicate pairs...")
    print("=" * 80)
    
    # Find specific duplicate pairs
    duplicates = deduplicator.find_duplicates(corpus)
    
    print(f"\nFound {len(duplicates)} duplicate pairs:")
    for idx1, idx2, sim in duplicates:
        print(f"\nSimilarity: {sim:.2%}")
        print(f"  [{idx1}] {corpus[idx1][:50]}...")
        print(f"  [{idx2}] {corpus[idx2][:50]}...")
    
    print("\n" + "=" * 80)
    print("Jaccard Similarity Estimation")
    print("=" * 80)
    
    # Demonstrate similarity estimation
    doc1 = "The quick brown fox jumps over the lazy dog."
    doc2 = "The quick brown fox jumps over the lazy dog!"  # Same except punctuation
    
    sig1 = deduplicator.minhash.compute_signature(
        deduplicator.processor.create_shingles(doc1, k=5)
    )
    sig2 = deduplicator.minhash.compute_signature(
        deduplicator.processor.create_shingles(doc2, k=5)
    )
    
    sim = deduplicator.minhash.similarity(sig1, sig2)
    print(f"\nDocument 1: {doc1}")
    print(f"Document 2: {doc2}")
    print(f"Estimated Jaccard similarity: {sim:.2%}")


if __name__ == "__main__":
    demo()
