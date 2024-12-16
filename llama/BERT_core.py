import torch
from transformers import BertModel, BertTokenizer


class BERTLayerActivationExtractor:
    def __init__(self, model_name='bert-base-uncased'):
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

    def extract_word_activation(self, sentence, target_word):
        """
        Extract layer activations for a specific word in a sentence

        Args:
            sentence (str): Full input sentence
            target_word (str): Word to extract activations for

        Returns:
            torch.Tensor: Activation vector for the target word
        """
        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True)

        # Get token ids and convert to list
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Find the index of the target word (accounting for special tokens)
        try:
            # Find all occurrences of the target word
            word_indices = [
                i for i, token in enumerate(tokens)
                if token.lower().replace('##', '') == target_word.lower()
            ]

            if not word_indices:
                raise ValueError(f"Target word '{target_word}' not found in sentence")

            # If multiple occurrences, use the first one
            word_index = word_indices[0]
        except Exception as e:
            print(f"Error finding target word: {e}")
            return None

        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Get all hidden states (layers)
        hidden_states = outputs.hidden_states
        # for layer in range(13):
        #     print(hidden_states[layer].shape)

        activations = [hidden_states[i][0][word_index] for i in range(len(hidden_states))]
        return activations


    def print_layer_info(self, sentence, target_word):
        """
        Print detailed information about the layer activations

        Args:
            sentence (str): Full input sentence
            target_word (str): Word to extract activations for
        """
        activations = self.extract_word_activation(sentence, target_word)

        if activations is not None:
            print(f"Activations for '{target_word}' at layer {self.target_layer}:")
            print(f"Shape: {activations.shape}")
            print(f"First 10 values: {activations[:10]}")
            print(f"Mean activation: {activations.mean().item()}")
            print(f"Standard deviation: {activations.std().item()}")


# Example usage
def main():
    # Create extractor for a specific layer (e.g., last layer)
    extractor = BERTLayerActivationExtractor(target_layer=-1)

    # Example sentence
    sentence = "The quick brown fox jumps over the lazy dog"
    target_word = "fox"

    # Extract and print layer activations
    extractor.print_layer_info(sentence, target_word)

    # If you want to get the raw activations for further processing
    activations = extractor.extract_word_activation(sentence, target_word)


if __name__ == "__main__":
    main()