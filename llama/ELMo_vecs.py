# import torch
from allennlp.commands.elmo import ElmoEmbedder
import spacy


class ElmoWordEmbedding:
    def __init__(self):
        # Initialize ELMo embedder
        self.elmo = ElmoEmbedder()
        # Initialize spaCy for tokenization
        self.nlp = spacy.load('en_core_web_sm')

    def get_word_embedding(self, sentence, target_word):
        """
        Get ELMo embeddings for a target word in a sentence.

        Args:
            sentence (str): Input sentence
            target_word (str): Target word to get embeddings for

        Returns:
            dict: Dictionary containing embeddings for each layer
        """
        # Tokenize the sentence
        tokens = [token.text for token in self.nlp(sentence)]

        # Find target word position
        try:
            target_positions = [i for i, word in enumerate(tokens) if word.lower() == target_word.lower()]
            if not target_positions:
                raise ValueError(f"Target word '{target_word}' not found in sentence")
        except ValueError as e:
            print(e)
            return None

        # Get ELMo embeddings for the sentence
        embeddings = self.elmo.embed_sentence(tokens)

        # Extract embeddings for each layer for the target word
        word_embeddings = {
            'character_layer': embeddings[0][target_positions[0]],  # Character-level layer
            'lstm_layer1': embeddings[1][target_positions[0]],  # First LSTM layer
            'lstm_layer2': embeddings[2][target_positions[0]]  # Second LSTM layer
        }

        return word_embeddings

    def get_embedding_shape(self):
        """Returns the shape of embeddings for each layer"""
        return {
            'character_layer': 512,
            'lstm_layer1': 1024,
            'lstm_layer2': 1024
        }


def main():
    # Example usage
    elmo_embedder = ElmoWordEmbedding()

    # Test with a sample sentence
    sentence = "The quick brown fox jumps over the lazy dog"
    target_word = "fox"

    embeddings = elmo_embedder.get_word_embedding(sentence, target_word)

    if embeddings:
        print(f"\nEmbeddings for word '{target_word}':")
        for layer, embedding in embeddings.items():
            print(f"\n{layer}:")
            print(f"Shape: {embedding.shape}")
            print(f"First few dimensions: {embedding[:5]}")


if __name__ == "__main__":
    main()