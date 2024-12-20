import torch
import numpy as np


def find_word_indices(input_ids, word_tokens):
    indices = []
    for i in range(len(input_ids) - len(word_tokens) + 1):
        if input_ids[i:i + len(word_tokens)].tolist() == word_tokens:
            indices.append(i)
    return indices

class BERTLayerActivationExtractor:
    def __init__(self, model_name='bert-base-uncased'):
        from transformers import BertModel, BertTokenizer
        """
        Initialize BERT model and tokenizer

        Args:
            model_name (str): Hugging Face model name
            target_layer (int): Layer index to extract activations from
                               (default is -1, the last layer)
        """
        # Load pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)

        # Set model to evaluation mode
        self.model.eval()

        # Validate and set target layer
        # self.target_layer = target_layer

    def extract_activations(self, sentence, target_words):
        """
        Extract layer activations for a specific word in a sentence

        Args:
            sentence (str): Full input sentence
            target_words (str): Word to extract activations for

        Returns:
            torch.Tensor: Activation vector for the target word
        """
        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True)

        # Get token ids and convert to list
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print(f'{sentence=}')
        print(f'\t{inputs=}')
        print(f'\t{tokens=}')

        word_idxs_all = []
        for word in target_words:
            word_token = self.tokenizer.encode(word, add_special_tokens=False)
            word_idxs = find_word_indices(inputs.input_ids[0], word_token)
            if len(word_idxs) < 1:
                raise ValueError(f"Target word '{word}' not found in sentence ({target_words=})")
            num_words = max(word_idxs) - min(word_idxs) + 1
            word_idxs_all.append((min(word_idxs), num_words))

        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Get all hidden states (layers)
        hidden_states = outputs.hidden_states
        activations_all = []
        for (word_idx, num_words) in word_idxs_all:
            idx_st = word_idx
            idx_end = word_idx + num_words
            activations = np.array([hidden_states[i][0][idx_st:idx_end] for i in range(len(hidden_states))])
            activations = np.nanmean(activations, axis=1)
            activations_all.append(activations)
        activations_all = np.array(activations_all)
        return activations_all



# Example usage
def main():
    # Create extractor for a specific layer (e.g., last layer)
    extractor = BERTLayerActivationExtractor()

    # Example sentence
    sentence = "The quick brown fox jumps over the lazy dog"
    target_word = "fox"

    # If you want to get the raw activations for further processing
    activations = extractor.extract_activations(sentence, target_word)
    print(f'{activations.shape=}')

if __name__ == "__main__":
    main()