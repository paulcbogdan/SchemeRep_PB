import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np



class LlamaActivationExtractor:
    def __init__(self, model_name='meta-llama/Llama-3-8b-hf'):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
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

    def _register_comprehensive_hooks(self):
        def attention_big_hook(layer_name):
            def attention_big_hook_(module, input, output):
                # print(module)
                # quit()

                hidden_states = input[0]
                # print(output[0].shape) # attention output, weighed by the attention weights

                # Compute query, key, value projections
                query_states = module.self_attn.q_proj(hidden_states)
                key_states = module.self_attn.k_proj(hidden_states)
                value_states = module.self_attn.v_proj(hidden_states)

                # Reshape and compute attention scores
                query_states = query_states.view(
                    query_states.size(0),
                    query_states.size(1),
                    module.self_attn.num_heads,
                    module.self_attn.head_dim
                )

                key_states = key_states.view(
                    key_states.size(0),
                    key_states.size(1),
                    # module.self_attn.num_key_value_groups,
                    module.self_attn.num_key_value_heads,
                    module.self_attn.head_dim,
                )
                key_states = key_states.repeat_interleave(
                    module.self_attn.num_key_value_groups, dim=2)

                # Compute attention scores
                attn_weights = torch.matmul(
                    query_states.transpose(1, 2),
                    key_states.transpose(1, 2).transpose(-1, -2)
                ) / (module.self_attn.head_dim ** 0.5)


                # attention_mask = torch.tril(torch.ones(hidden_states.size(1), hidden_states.size(1)))
                # Softmax to get attention probabilities
                # attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

                self.attention_act['input'][layer_name] = hidden_states
                self.attention_act['q_proj'][layer_name] = query_states
                self.attention_act['k_proj'][layer_name] = key_states
                self.attention_act['v_proj'][layer_name] = value_states
                self.attention_act['attn_output'][layer_name] = output[0] if isinstance(output, tuple) else output
                self.attention_act['attn_weights'][layer_name] = attn_weights
                # print(attn_weights.size())
                # quit()
            return attention_big_hook_

        def activation_hook(layer_name, dict_to_store_in, dict_to_store_out):
            def hook(module, input, output):
                dict_to_store_in[layer_name] = input[0]
                dict_to_store_out[layer_name] = output
            return hook

        self.hooks = []
        for name, module in self.model.named_modules():
            name_spl = name.split('.')
            if len(name_spl) < 2: continue
            if name_spl[-2] == 'layers':
                layer_num = int(name.split('.')[-1])
                hook = module.register_forward_hook(attention_big_hook(layer_num))
                self.hooks.append(hook)
            if name_spl[-2] == 'mlp':
                layer_num = int(name.split('.')[-3])
                hook = module.register_forward_hook(
                    activation_hook(layer_num, self.mlp_act_in[name_spl[-1]],
                                    self.mlp_act_out[name_spl[-1]]))
                self.hooks.append(hook)

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
            print(f'{word}, {word_token}')
            target_idx = self._find_word_indices(inputs.input_ids[0], word_token)

            if len(target_idx) == 0:
                variants = [f'{word.capitalize()}', f'{word.capitalize()}s', f'{word.capitalize()}es',
                            f' {word}s', f' {word}es']
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

        # Print activations
        for key, d in self.attention_act.items():
            if key == 'attn_weights':
                out['attn'][key] = self._extract_target_attn_weights(d, target_indices[0],
                                                                     target_indices[1])
            else:
                out['attn'][key] = self._extract_target_activations(d, target_indices)
        for key, d in self.mlp_act_in.items():
            out['mlp_in'][key] = self._extract_target_activations(d, target_indices)
        for key, d in self.mlp_act_out.items():
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
    extractor = LlamaActivationExtractor(r'baffo32/decapoda-research-llama-7B-hf')

    # extractor = LlamaActivationExtractor('meta-llama/Llama-2-7b-hf')
    # extractor = LlamaActivationExtractor(r'meta-llama/Llama-3.2-3b-Instruct')
    # extractor = LlamaActivationExtractor(r'meta-llama/Llama-3.2-3b')

    # extractor = LlamaActivationExtractor(r'meta-llama/Llama-3.2-1b')
    # extractor = LlamaActivationExtractor(r'meta-llama/Llama-3.1-70b')

    extractor._register_comprehensive_hooks()

    sentence = "The quick brown fox jumps over the lazy dog."
    # sentence = "At the bank, a bench"
    target_word = "fox"


    try:
        # activations, tokens = extractor.extract_activations(sentence, target_word)
        # results = extractor.extract_activations(sentence, ['bank', 'bench'])
        results = extractor.extract_activations(sentence, ['fox', 'over'])

        # print(results)


    finally:
        extractor.cleanup()




if __name__ == "__main__":
    main()