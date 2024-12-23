import torch

import numpy as np

from llama.BERT_core import find_word_indices


class DetailedSimCSEEmbedder:
    def __init__(self, model_name='princeton-nlp/sup-simcse-bert-base-uncased'):
        from transformers import AutoModel, AutoTokenizer
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

    def extract_activations(self, sentence, target_words):
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
        hidden_states = outputs.hidden_states
        detailed_embeddings = {}
        tokens = self.tokenizer.tokenize(sentence)

        word_idxs_all = []
        for word in target_words:
            word_token = self.tokenizer.encode(word, add_special_tokens=False)
            word_idxs = find_word_indices(encoded_input.input_ids[0], word_token)
            if len(word_idxs) < 1:
                raise ValueError(f"Target word '{word}' not found in sentence ({target_words=})")
            num_words = max(word_idxs) - min(word_idxs) + 1
            word_idxs_all.append((min(word_idxs), num_words))


        activations_all = []
        for (word_idx, num_words) in word_idxs_all:
            idx_st = word_idx
            idx_end = word_idx + num_words
            # TODO: can change this to no longer average...
            activations = np.array([hidden_states[i].cpu().numpy()[0, idx_st:idx_end, :]
                                    for i in range(len(hidden_states))])
            activations = np.nanmean(activations, axis=1) # average across idxs of a given word
            # print(activations.shape)
            activations_all.append(activations)
        activations_all = np.array(activations_all)
        return activations_all


# Example usage
if __name__ == "__main__":
    embedder = DetailedSimCSEEmbedder()

    sentence = "The quick brown fox jumps over the lazy dog."
    words_to_embed = ["fox", "dog"]

    detailed_embs = embedder.extract_activations(sentence, words_to_embed)


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