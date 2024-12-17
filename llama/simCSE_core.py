import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np


class SimCSEEmbedder:
    def __init__(self, model_name='princeton-nlp/sup-simcse-bert-base-uncased'):
        """
        Initialize SimCSE embedder with a pre-trained model.

        Args:
            model_name (str): Hugging Face model path for SimCSE embeddings.
                Default is the supervised SimCSE BERT base model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on model output.

        Args:
            model_output (torch.Tensor): Model's last hidden state
            attention_mask (torch.Tensor): Attention mask for the input

        Returns:
            torch.Tensor: Pooled sentence embedding
        """
        # Extract last hidden states
        token_embeddings = model_output.last_hidden_state

        # Create mask to ignore padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum masked embeddings and divide by number of tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def get_sentence_embedding(self, sentence):
        """
        Get embedding for an entire sentence.

        Args:
            sentence (str): Input sentence

        Returns:
            numpy.ndarray: Sentence embedding
        """
        # Tokenize sentence
        encoded_input = self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        # Get model output
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Mean pooling
        sentence_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])

        return sentence_embedding.cpu().numpy().flatten()

    def get_word_embeddings(self, sentence, words):
        """
        Get embeddings for specific words in a sentence.

        Args:
            sentence (str): Input sentence
            words (list): List of words to extract embeddings for

        Returns:
            dict: Mapping of words to their embeddings
        """
        # Tokenize sentence
        encoded_input = self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        # Get model output
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Get token-level embeddings
        token_embeddings = model_output.last_hidden_state.squeeze()

        # Get tokenized words
        tokens = self.tokenizer.tokenize(sentence)

        # Find word embeddings
        word_embeddings = {}
        for target_word in words:
            # Find all indices of the word (handling subword tokenization)
            word_token_indices = [
                i for i, token in enumerate(tokens)
                if target_word.lower() in token.lower()
            ]

            if word_token_indices:
                # Average embeddings for all matching token indices
                avg_embedding = token_embeddings[word_token_indices].mean(dim=0).cpu().numpy()
                word_embeddings[target_word] = avg_embedding

        return word_embeddings


# Example usage
if __name__ == "__main__":
    # Initialize embedder
    embedder = SimCSEEmbedder()

    # Example sentence and words
    sentence = "The quick brown fox jumps over the lazy dog."
    words_to_embed = ["fox", "dog", "quick"]

    # Get sentence embedding
    sentence_emb = embedder.get_sentence_embedding(sentence)
    print("Sentence Embedding Shape:", sentence_emb.shape)

    # Get word embeddings
    word_embs = embedder.get_word_embeddings(sentence, words_to_embed)
    for word, emb in word_embs.items():
        print(f"{word} Embedding Shape: {emb.shape}")