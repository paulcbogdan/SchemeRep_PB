from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.devereux_neuron import do_deve_neuron
import matplotlib.pyplot as plt

class ActivationModifier:
    def __init__(self, model_name,
                 activation_adjustments):
        """
        Initialize the activation modifier.

        Args:
            model_name: Name or path of the model
            layer_nums: List of layer numbers to modify
            activation_adjustments: Dict mapping layer numbers to (neuron_idx, adjustment_value) pairs
        """
        # self.model = AutoModelForCausalLM.from_pretrained(model_name,
        #                                                   torch_dtype=torch.float16,
        #                                                   # device_map='auto',
        #                                                   )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.layer_nums = list(activation_adjustments.keys())
        self.activation_adjustments = activation_adjustments
        self.hooks = []
        self._register_hooks()

    def _activation_hook(self, layer_num):
        def hook(module, input, output):
            # Assuming output is a tensor or tuple of tensors
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(input, tuple):
                input = input[0]

            # Apply the adjustments for this layer
            if layer_num in self.activation_adjustments:
                for neuron_idx, adjustment in self.activation_adjustments[layer_num]:
                    input[:, :, neuron_idx] += adjustment
                    # output[:, :, neuron_idx] += adjustment

            return output

        return hook

    def _register_hooks(self):
        # Get all transformer layers
        for name, module in self.model.named_modules():
            # print(module)
            # Adjust this pattern based on your specific model architecture
            print(name)

            if "layers" in name and any(f".{num}." in name for num in self.layer_nums):
                # print('test')
                # This assumes we're hooking into the output of the MLP layer
                # Adjust the pattern based on your model's architecture
                # if "mlp" in name and "output" in name:
                if 'mlp.gate_proj' in name:
                    layer_num = int(name.split(".")[2])  # Adjust split pattern if needed
                    hook = module.register_forward_hook(self._activation_hook(layer_num))
                    self.hooks.append(hook)


    def generate(self, prompt, max_length=50, **kwargs):
        """
        Generate text with modified activations.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate with modified activations
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.001,
            **kwargs
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def cleanup(self):
        """
        Remove all hooks
        """
        for hook in self.hooks:
            hook.remove()

def get_deve_neuron_t_signif():
    t_vals = pickle_wrap(do_deve_neuron, easy_override=True)
    return t_vals

# Example usage
if __name__ == "__main__":
    # Example: Modify activations in layers 5 and 8
    # adjustments = {
    #     5: [(100, 5000), (200, -0.3)],  # Adjust neurons 100 and 200 in layer 5
    #     8: [(150, 10000.7)],  # Adjust neuron 150 in layer 8
    #     15: [(150, 10000.7)]  # Adjust neuron 150 in layer 8
    # }


    adjustments = defaultdict(list)


    num_mods = 0
    t_vals = get_deve_neuron_t_signif()
    for layer in range(t_vals.shape[0]):
        for neuron in range(t_vals.shape[1]):
            if t_vals[layer, neuron] > 2:
                adjustments[layer].append((neuron, 0.3))
                num_mods += 1
            elif t_vals[layer, neuron] < -2:
                adjustments[layer].append((neuron, -0.3))
                num_mods += 1
    print(f'Modified {num_mods} neurons')

    # -Instruct
    modifier = ActivationModifier(
        model_name="meta-llama/Llama-3.2-1b",
        # layer_nums=[15],
        activation_adjustments=adjustments
    )

    # Generate text with modified activations
    prompt = "This circle is colored"
    output = modifier.generate(prompt, max_length=20)
    print(output)

    # Clean up hooks when done
    modifier.cleanup()
