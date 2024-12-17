import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np


class DetailedSimCSEEmbedder:
    def __init__(self, model_name='princeton-nlp/sup-simcse-bert-base-uncased'):
        """
        Initialize detailed SimCSE embedder with full layer access.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Extract configuration details
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

    def get_detailed_embeddings(self, sentence, words):
        """
        Extract detailed embeddings across all layers for specific words.

        Returns:
        - Detailed embedding dictionary with:
          * token-level embeddings for each layer
          * contextual representations
          * raw hidden states
        """
        # Tokenize sentence
        encoded_input = self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        # Get full model output with all hidden states
        with torch.no_grad():
            outputs = self.model(
                **encoded_input,
                output_hidden_states=True  # Crucial for accessing all layer embeddings
            )

        # Extract all hidden states
        all_hidden_states = outputs.hidden_states

        # Detailed embedding extraction
        detailed_embeddings = {}
        tokens = self.tokenizer.tokenize(sentence)

        for target_word in words:
            # Find token indices for the word
            word_token_indices = [
                i for i, token in enumerate(tokens)
                if target_word.lower() in token.lower()
            ]

            if word_token_indices:
                word_details = {
                    'layer_embeddings': [],  # Embeddings for each layer
                    'layer_details': []  # Additional layer-wise information
                }

                # Extract embeddings for each layer
                for layer_idx, layer_hidden_states in enumerate(all_hidden_states):
                    # Get embeddings for this layer
                    layer_embeddings = layer_hidden_states.squeeze()
                    word_layer_emb = layer_embeddings[word_token_indices].mean(dim=0)

                    word_details['layer_embeddings'].append(word_layer_emb.cpu().numpy())

                    # Optional: Add some layer-wise statistics
                    word_details['layer_details'].append({
                        'layer': layer_idx,
                        'embedding_mean': word_layer_emb.mean().item(),
                        'embedding_std': word_layer_emb.std().item(),
                    })

                detailed_embeddings[target_word] = word_details

        return {
            'embeddings': detailed_embeddings,
            'num_layers': self.num_hidden_layers,
            'hidden_size': self.hidden_size
        }


# Example usage
if __name__ == "__main__":
    embedder = DetailedSimCSEEmbedder()

    sentence = "The quick brown fox jumps over the lazy dog."
    words_to_embed = ["fox", "dog"]

    detailed_embs = embedder.get_detailed_embeddings(sentence, words_to_embed)


    # Demonstrate layer-wise information
    for word, word_details in detailed_embs['embeddings'].items():
        print(f"\nWord: {word}")
        print(f"Number of layers: {len(word_details['layer_embeddings'])}")
        print(np.array(word_details['layer_embeddings']).shape)
        print(f"Embedding size per layer: {word_details['layer_embeddings'][0].shape}")

        # # Print some layer details
        # print("\nLayer-wise Statistics:")
        # for layer_detail in word_details['layer_details']:
        #     print(f"Layer {layer_detail['layer']}:")
        #     print(f"  Mean: {layer_detail['embedding_mean']}")
        #     print(f"  Std Dev: {layer_detail['embedding_std']}")