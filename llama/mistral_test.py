import torch
from torch.onnx.symbolic_opset9 import tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from transformers import BitsAndBytesConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# from transformers import MistralForCausalLM, MistralTokenizer

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    # from: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]

def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)

class LlamaActivationExtractor:
    def __init__(self, model_name='meta-llama/Llama-3-8b-hf',
                 ignore_attn=None, ignore_mlp_in=None, ignore_mlp_out=None):
        #
        # if '70b' or '7b' in model_name:
        #     bnb_config = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype="float16"
        #     )
        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         r'C:\Users\paulc\.cache\huggingface\hub\models--meta-llama--Llama-3.3-70b-Instruct\snapshots\5825c9120fc701a0b7d9a30d61005f2a09466b74',
        #         device_map="auto",
        #         quantization_config=bnb_config,
        #         offload_folder=r'C:\PycharmProjects\SchemeRep\llama\offload'
        #
        #     )
        # else:

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name

        # Specialized activation containers
        self.mlp_act_in = {'gate_proj': {}, 'up_proj': {}, 'down_proj': {},
                           'act_fn': {}}
        self.mlp_act_out = {'gate_proj': {}, 'up_proj': {}, 'down_proj': {},
                            'act_fn': {}}
        self.attention_act = {'q_proj': {}, 'k_proj': {}, 'v_proj': {},
                              'attn_output': {}, 'attn_weights': {},
                              'input': {}}

        # Store hook handles for cleanup
        self.hooks = []
        self.ignore_attn = ignore_attn
        self.ignore_mlp_in = ignore_mlp_in
        self.ignore_mlp_out = ignore_mlp_out

    def _register_comprehensive_hooks(self):

        def attention_big_hook(layer_name):
            def attention_big_hook_(module, input, output):

                hidden_states = input[0]
                # Compute query, key, value projections
                query_states = module.self_attn.q_proj(hidden_states)
                key_states = module.self_attn.k_proj(hidden_states)
                value_states = module.self_attn.v_proj(hidden_states)

                num_heads = 32
                num_key_value_heads = 8
                head_dim = 4096 // num_heads
                num_kvh_groups = 4

                # print(query_states.size())

                query_states = query_states.view(
                    query_states.size(0),
                    query_states.size(1),
                    num_heads,
                    head_dim
                )

                key_states = key_states.view(
                    key_states.size(0),
                    key_states.size(1),
                    num_key_value_heads,
                    head_dim
                )

                value_states = value_states.view(
                    value_states.size(0),
                    value_states.size(1),
                    num_key_value_heads,
                    head_dim
                )

                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                rope_config = {              # RoPE frequency scaling
                    "factor": 32.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_context_length": 8192,
                }
                mask, cos, sin = SharedBuffers.get_buffers(192,#8192,#48,
                                                           module.self_attn.head_dim, 500_000.0,
                                                           rope_config, torch.bfloat16)
                key_states = compute_rope(key_states, cos.to(self.model.device),
                                          sin.to(self.model.device))
                query_states = compute_rope(query_states, cos.to(self.model.device),
                                            sin.to(self.model.device))

                key_states = key_states.repeat_interleave(
                    num_kvh_groups, dim=1)

                value_states = value_states.repeat_interleave(
                    num_kvh_groups, dim=1)


                attn_scores = (query_states @ key_states.transpose(2, 3) /
                               (module.self_attn.head_dim ** 0.5))

                self.attention_act['input'][layer_name] = hidden_states
                self.attention_act['v_proj'][layer_name] = value_states
                self.attention_act['attn_weights'][layer_name] = attn_scores

            return attention_big_hook_

        def activation_hook(layer_name, dict_to_store_in, dict_to_store_out):
            def hook(module, input, output):
                dict_to_store_in[layer_name] = input[0]
                dict_to_store_out[layer_name] = output
                # print(f'{dict_to_store_out=}')
            return hook

        # def activation_hook_rotary(layer_name, dict_to_store_in, dict_to_store_out):
        #     def hook(module, input, output):
        #         # dict_to_store_in[layer_name] = input[0]
        #         dict_to_store_out[layer_name] = output
        #     return hook

        self.hooks = []
        for name, module in self.model.named_modules():
            name_spl = name.split('.')
            print(f'{name=}')
            if len(name_spl) < 2: continue
            if name_spl[-2] == 'self_attn':
                if name_spl[-1] == 'o_proj':
                    layer_num = int(name.split('.')[-3])
                    hook = module.register_forward_hook(activation_hook(
                        layer_num, {}, self.attention_act['attn_output']))
                    self.hooks.append(hook)
            if name_spl[-2] == 'layers':
                layer_num = int(name.split('.')[-1])
                hook = module.register_forward_hook(attention_big_hook(layer_num))
                self.hooks.append(hook)
            if name_spl[-2] == 'mlp':
                # continue
                layer_num = int(name.split('.')[-3])
                hook = module.register_forward_hook(
                    activation_hook(layer_num, self.mlp_act_in[name_spl[-1]],
                                    self.mlp_act_out[name_spl[-1]]))
                self.hooks.append(hook)
            # if name_spl[]

        return self

    def extract_activations(self, sentence, target_words):
        # Reset activations
        for key, d in self.attention_act.items():
            d.clear()
        for key, d in self.mlp_act_in.items():
            d.clear()
        for key, d in self.mlp_act_out.items():
            d.clear()
        print(f'llama sentence: {sentence=}')
        # Tokenize and process
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.model.device)
        # Decode tokens for debugging
        decoded_tokens = [self.tokenizer.decode(token) for token in inputs.input_ids[0]]
        print(f'\t split: {decoded_tokens}')
        print(f'\t tokens: {inputs.input_ids[0]}')

        if isinstance(target_words, str):
            target_words = [target_words]
        target_words_tokens = []
        target_indices = []
        for word in target_words:
            if r'meta-llama/Llama-2-7b' in self.model_name:
                word_token = self.tokenizer.encode(word, add_special_tokens=False)
            else:
                word_token = self.tokenizer.encode(f' {word}', add_special_tokens=False)
            print(f'{word}: {word_token}')
            target_idx = self._find_word_indices(inputs.input_ids[0], word_token)

            if len(target_idx) == 0:
                variants = [f'{word.capitalize()}', f'{word.capitalize()}s', f'{word.capitalize()}es',
                            f' {word}s', f' {word}es', f' {word}\'s', f' {word.capitalize()}',
                            f'{word}', ' ' + word.replace('us', 'i')]
                for v in variants:
                    word_token = self.tokenizer.encode(f'{v}', add_special_tokens=False)
                    target_idx = self._find_word_indices(inputs.input_ids[0], word_token)
                    if len(target_idx) > 0:
                        print(f'Found: {v}')
                        break
            if len(target_idx) == 0:
                print(f'BAD!! Word not found: {word}')
            target_indices.append((target_idx[0], len(word_token)))
            target_words_tokens.append(word_token)

        assert len(target_indices) > 0, (f"No target words ({target_words}) found in "
                                         f"the input sentence: {sentence}")
        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)

        out = {'attn': {}, 'mlp_in': {}, 'mlp_out': {},
               'words': target_words,
               'word_tokens': target_words_tokens,
               'word_indices': target_indices,
               'sentence': sentence
               }

        # print(self.attention_act)
        # quit()
        # Print activations
        for key, d in self.attention_act.items():
            if 'q_proj' in key: continue
            if 'k_proj' in key: continue
            if key == 'attn_weights' and len(target_indices) > 1:
                out['attn'][key] = self._extract_target_attn_weights(d, target_indices[0],
                                                                     target_indices[1])
            else:
                out['attn'][key] = self._extract_target_activations(d, target_indices)
        for key, d in self.mlp_act_in.items():
            if 'down' in key: continue
            if 'up' in key: continue
            if 'act_fn' in key: continue
            out['mlp_in'][key] = self._extract_target_activations(d, target_indices)
        for key, d in self.mlp_act_out.items():
            if 'gate' in key: continue
            if 'up' in key: continue
            out['mlp_out'][key] = self._extract_target_activations(d, target_indices)

        # if r'meta-llama/Llama-2-7b' in self.model_name:
        # for outer, d_outer in out.items():
        #     if outer not in ['attn', 'mlp_in', 'mlp_out']: continue
        #     for inner, d_inner in d_outer.items():
        #         for key, val in d_inner.items():
        #             print(f'{outer}, {inner}, {key}')
        #             try:
        #                 d_outer[inner][0] = val[0].cpu().numpy()
        #                 d_outer[inner][1] = val[1].cpu().numpy()
        #                 print(f'{type(val[0])=}')
        #             except:
        #                 pass


        return out

    def _find_word_indices(self, input_ids, word_tokens):
        indices = []
        for i in range(len(input_ids) - len(word_tokens) + 1):
            if input_ids[i:i + len(word_tokens)].tolist() == word_tokens:
                indices.append(i)
        return indices


    def _extract_target_activations(self, activations_dict, target_indices):
        result = {}
        for i, (layer, activation) in enumerate(activations_dict.items()):
            module_activations = []
            for (idx, num) in target_indices:
                try:
                    module_activations.append(activation[0, idx:idx + num].
                                              numpy(force=True))
                except Exception as e:
                    print(f"Extraction error: {e}")
            result[layer] = module_activations # can't numpy array because the number of idxs in each word may differ
        return result

    def _extract_target_attn_weights(self, activations_dict, idx0, idx1):
        result = {}
        st0 = idx0[0]
        end0 = idx0[0] + idx0[1]
        st1 = idx1[0]
        end1 = idx1[0] + idx1[1]
        for i, (layer, activation) in enumerate(activations_dict.items()):
            result[layer] = []
            # res = extractor.extract_activations(sentence, [obj, scn], )
            #   thus obj is idx0, scn is idx1

            result[layer].append(activation[0, :, st0:end0, st1:end1].numpy(
                force=True).transpose(1, 2, 0)) # idx0 -> idx1
            result[layer].append(activation[0, :, st1:end1, st0:end0].numpy(
                force=True).transpose(1, 2, 0)) # idx1 -> idx0
        return result

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


def main():
    # extractor = LlamaActivationExtractor(r'baffo32/decapoda-research-llama-7B-hf')

    # extractor = LlamaActivationExtractor('meta-llama/Llama-2-7b-hf')
    # extractor = LlamaActivationExtractor(r'meta-llama/Llama-3.2-3b-Instruct')
    # extractor = LlamaActivationExtractor(r'meta-llama/Llama-3.2-3b')

    extractor = LlamaActivationExtractor(r'mistralai/Mistral-7b-v0.3')
    # extractor = LlamaActivationExtractor(r'meta-llama/Llama-3.1-70b')

    extractor._register_comprehensive_hooks()

    sentence = "The quick brown fox jumps over the lazy dog."
    # sentence = "At the bank, a bench"
    target_word = "fox"


    try:
        # activations, tokens = extractor.extract_activations(sentence, target_word)
        # results = extractor.extract_activations(sentence, ['bank', 'bench'])
        results = extractor.extract_activations(sentence, ['fox', 'over'])



    finally:
        extractor.cleanup()




if __name__ == "__main__":
    main()