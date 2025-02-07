#!/usr/bin/env python
import argparse
import torch
from typing import Dict, Any, List, Tuple
from transformers import LlamaConfig, AutoModelForCausalLM, AutoTokenizer
from torchtitan.models.llama import Transformer
from torchtitan.models.llama.model import ModelArgs


def permute(w, n_heads, dim1, dim2):
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def build_model(config):
    model_cls = Transformer
    # set the model configs from training inputs:
    model_config = ModelArgs(
        dim=config["hidden_size"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        vocab_size=config["vocab_size"],
    )
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)
    return model


def to_huggingface(states: Dict[str, Any], config: Dict):
    # step 1: create a config file containing the model architecture
    hf_config = LlamaConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_dim"],
        num_hidden_layers=config["n_layers"],
        num_attention_heads=config["n_heads"],
        num_key_value_heads=config["n_kv_heads"],
        hidden_act=config["activation"],
        max_position_embeddings=config["max_seq_len"],
        initializer_range=config["initializer_range"],
        rope_theta=config["rope_theta"],
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config)
    # step 2: load the model weights, translating them accordingly
    new_state_dict = {}
    with torch.inference_mode():
        states_dicts = states["model"]
        states_dicts = {k: v.contiguous() for k, v in states_dicts.items()}
        new_state_dict["model.embed_tokens.weight"] = states_dicts["tok_embeddings.weight"]
        new_state_dict["model.norm.weight"] = states_dicts["norm.weight"]
        new_state_dict["lm_head.weight"] = states_dicts["output.weight"]
        dims_per_head = hf_config.hidden_size // hf_config.num_attention_heads

        for i in range(hf_config.num_hidden_layers):
            new_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = permute(
                w=states_dicts[f"layers.{i}.attention.wq.weight"],
                n_heads=hf_config.num_attention_heads,
                dim1=hf_config.hidden_size,
                dim2=hf_config.hidden_size,
            )
            new_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = permute(
                w=states_dicts[f"layers.{i}.attention.wk.weight"],
                n_heads=hf_config.num_key_value_heads,
                dim1=dims_per_head * hf_config.num_key_value_heads,
                dim2=hf_config.hidden_size,
            )
            new_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = states_dicts[
                f"layers.{i}.attention.wv.weight"
            ]
            new_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = states_dicts[
                f"layers.{i}.attention.wo.weight"
            ]
            new_state_dict[f"model.layers.{i}.input_layernorm.weight"] = states_dicts[
                f"layers.{i}.attention_norm.weight"
            ]
            new_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = states_dicts[
                f"layers.{i}.ffn_norm.weight"
            ]
            new_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = states_dicts[
                f"layers.{i}.feed_forward.w1.weight"
            ]
            new_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = states_dicts[
                f"layers.{i}.feed_forward.w2.weight"
            ]
            new_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = states_dicts[f"layers.{i}.feed_forward.w3.weight"]

        model.load_state_dict(new_state_dict, strict=True, assign=True)
        model.eval()
        return model, hf_config


def get_divisors(n: int) -> List[int]:
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


def infer_heads_and_kv(states: Dict[str, torch.Tensor], hidden_size: int) -> Tuple[int, int]:
    """
    Infers the number of attention heads and key-value heads using the query and key weight shapes.
    The assumption is that the query weight has shape (hidden_size, hidden_size) and its permutation
    requires that hidden_size is divisible by (2 * n_heads), so n_heads must be a divisor of (hidden_size//2).
    For each candidate n_heads, we compute dims_per_head = hidden_size // (2 * candidate) and then check the key
    projection weight's shape: it should be (n_kv_heads * dims_per_head, hidden_size).
    We accept as candidate those for which n_kv_heads equals candidate.
    """
    candidates = []
    possible_values = get_divisors(hidden_size)
    # Check for candidates that ensure hidden_size % (2*n_heads)==0
    for candidate in possible_values:
        if hidden_size % (2 * candidate) == 0:
            dims_per_head = hidden_size // (2 * candidate)
            key_shape = states[
                "layers.0.attention.wk.weight"
            ].shape  # expected shape: (n_kv_heads * dims_per_head, hidden_size)
            if key_shape[0] % dims_per_head != 0:
                continue
            n_kv_heads_candidate = key_shape[0] // dims_per_head
            # Accept candidate only if n_kv_heads equals candidate.
            if n_kv_heads_candidate == candidate:
                candidates.append((candidate, n_kv_heads_candidate))
    if len(candidates) != 1:
        print(
            f"Warning: Found multiple candidates, using the first one. Please specify the number of heads!\ncandidates = {candidates}"
        )
    return candidates[0]  # returns (n_heads, n_kv_heads)


def infer_config_from_checkpoint(state_dict: Dict[str, Any], max_seq_len: int, n_kv_heads_override: int = None, n_heads_override: int = None, rope_theta: int = None) -> Dict:
    states = state_dict["model"]
    # Infer hidden_size from the token embedding weight (dimensions: vocab_size x hidden_size)
    vocab_size, hidden_size = states["tok_embeddings.weight"].shape
    # Count unique transformer layers using keys like "layers.{i}.attention.wq.weight"
    layer_indices = set()
    for key in states.keys():
        if key.startswith("layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
    if not layer_indices:
        raise ValueError("No transformer layer keys found in checkpoint!")
    n_layers = max(layer_indices) + 1
    n_kv_heads = n_kv_heads_override if n_kv_heads_override is not None else infer_heads_and_kv(states, hidden_size)[1]
    n_heads = n_heads_override if n_heads_override is not None else infer_heads_and_kv(states, hidden_size)[0]
    # Infer intermediate dimension from one of the feed-forward weights:
    # Expected shape for feed_forward.w1.weight is (intermediate_dim, hidden_size)
    intermediate_dim = states["layers.0.feed_forward.w1.weight"].shape[0]

    config = {
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "max_seq_len": max_seq_len,
        "vocab_size": vocab_size,
        "intermediate_dim": intermediate_dim,
        "activation": "silu",
        "initializer_range": 0.02,
        "rope_theta": 10000 if rope_theta is None else rope_theta,
        "norm_eps": 1e-05,
    }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TorchTITAN checkpoint to Hugging Face format.")
    parser.add_argument("--input", type=str, required=True, help="Path to the TorchTITAN .pt checkpoint file.")
    parser.add_argument("--output", type=str, required=True, help="Output folder path for the Hugging Face model.")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b", help="Which tokenizer to use.")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--n_kv_heads", type=int, default=None, help="Override number of key-value heads if provided.")
    parser.add_argument("--n_heads", type=int, default=None, help="Override number of heads if provided.")
    parser.add_argument("--rope_theta", type=int, default=None, help="Override rope theta if provided.")

    args = parser.parse_args()

    checkpoint_path = args.input
    output_path = args.output

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    # Infer configuration parameters, passing max_seq_len and n_kv_heads override if provided.
    inferred_config = infer_config_from_checkpoint(state_dict, args.max_seq_len, args.n_kv_heads, args.n_heads, args.rope_theta)

    print("Inferred configuration:")
    for key, value in inferred_config.items():
        print(f"  {key}: {value}")

    print("Building model...")
    model = build_model(inferred_config)
    model.to_empty(device="cpu")
    print("Converting model to Hugging Face format...")
    model, hf_config = to_huggingface(state_dict, inferred_config)
    print(f"Saving model and configuration to {output_path}...")
    model.save_pretrained(output_path)
    hf_config.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.save_pretrained(output_path)
