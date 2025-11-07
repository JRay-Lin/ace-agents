"""
Utility functions for the ACE framework.

This module provides helper functions for semantic similarity computation,
text processing, and other common operations.
"""

from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import logging


class SemanticSimilarity:
    """
    @brief Computes semantic similarity between text strings.

    This class uses sentence transformers to compute semantic embeddings
    and cosine similarity between texts.

    @param model_name Name of the sentence transformer model
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        @brief Initialize the semantic similarity calculator.

        @param model_name Sentence transformer model name
        """
        self.logger = logging.getLogger(__name__ + ".SemanticSimilarity")
        self.logger.info(f"Loading sentence transformer model: {model_name}")

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        @brief Compute cosine similarity between two texts.

        @param text1 First text
        @param text2 Second text
        @return Similarity score between 0 and 1
        """
        embeddings = self.model.encode([text1, text2])
        similarity = self._cosine_similarity(embeddings[0], embeddings[1])
        return float(similarity)

    def compute_pairwise_similarities(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        @brief Compute pairwise similarities for a list of texts.

        @param texts List of text strings
        @return NxN similarity matrix
        """
        if len(texts) == 0:
            return np.array([])

        embeddings = self.model.encode(texts)
        similarities = np.zeros((len(texts), len(texts)))

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities[i, j] = sim
                similarities[j, i] = sim

        return similarities

    def find_similar_pairs(
        self,
        texts: List[str],
        threshold: float = 0.9
    ) -> List[Tuple[int, int, float]]:
        """
        @brief Find pairs of texts with similarity above threshold.

        @param texts List of text strings
        @param threshold Similarity threshold
        @return List of (index1, index2, similarity) tuples
        """
        similarities = self.compute_pairwise_similarities(texts)
        similar_pairs = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if similarities[i, j] >= threshold:
                    similar_pairs.append((i, j, float(similarities[i, j])))

        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)

        return similar_pairs

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        @brief Compute cosine similarity between two vectors.

        @param vec1 First vector
        @param vec2 Second vector
        @return Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
