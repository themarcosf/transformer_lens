import einops

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_distilbertForSequenceClassification_weights(distilbert, cfg: HookedTransformerConfig):
    return distilbert.state_dict()
    # embeddings = distilbert.distilbert.embeddings
    # state_dict = {
    #     "embed.W_E": embeddings.word_embeddings.weight,
    #     "pos_embed.W_pos": embeddings.position_embeddings.weight,
    #     "embed.ln.w": embeddings.LayerNorm.weight, #this
    #     "embed.ln.b": embeddings.LayerNorm.bias, # this
    # }

    # for l in range(cfg.n_layers):
    #     block = distilbert.distilbert.transformer.layer[l]

    #     # Attention block
    #     state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
    #         block.attention.q_lin.weight, "(i h) m -> i m h", i=cfg.n_heads
    #     )
    #     state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
    #         block.attention.q_lin.bias, "(i h) -> i h", i=cfg.n_heads
    #     )
    #     state_dict[f"blocks.{l}.attn.W_K"] = einops.rearrange(
    #         block.attention.k_lin.weight, "(i h) m -> i m h", i=cfg.n_heads
    #     )
    #     state_dict[f"blocks.{l}.attn.b_K"] = einops.rearrange(
    #         block.attention.q_lin.bias, "(i h) -> i h", i=cfg.n_heads
    #     )
    #     state_dict[f"blocks.{l}.attn.W_V"] = einops.rearrange(
    #         block.attention.v_lin.weight, "(i h) m -> i m h", i=cfg.n_heads
    #     )
    #     state_dict[f"blocks.{l}.attn.b_V"] = einops.rearrange(
    #         block.attention.v_lin.bias, "(i h) -> i h", i=cfg.n_heads
    #     )
    #     state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
    #         block.attention.out_lin.weight, "m (i h) -> i h m", i=cfg.n_heads
    #     )
    #     state_dict[f"blocks.{l}.attn.b_O"] = block.attention.out_lin.bias

    #     # Intermediate LayerNorm
    #     state_dict[f"blocks.{l}.ln1.w"] = block.sa_layer_norm.weight
    #     state_dict[f"blocks.{l}.ln1.b"] = block.sa_layer_norm.bias

    #     # MLP block (Feed Forward)
    #     state_dict[f"blocks.{l}.mlp.W_in"] = einops.rearrange(
    #         block.ffn.lin1.weight, "mlp model -> model mlp"
    #     )
    #     state_dict[f"blocks.{l}.mlp.b_in"] = block.ffn.lin1.bias
    #     state_dict[f"blocks.{l}.mlp.W_out"] = einops.rearrange(
    #         block.ffn.lin2.weight, "model mlp -> mlp model"
    #     )
    #     state_dict[f"blocks.{l}.mlp.b_out"] = block.ffn.lin2.bias

    #     # Output LayerNorm
    #     state_dict[f"blocks.{l}.ln2.w"] = block.output_layer_norm.weight
    #     state_dict[f"blocks.{l}.ln2.b"] = block.output_layer_norm.bias

    # pre_classifier = distilbert.pre_classifier
    # state_dict["pre_classifier.W"] = pre_classifier.weight.T # this
    # state_dict["pre_classifier.b"] = pre_classifier.bias # this

    # classifier = distilbert.classifier
    # state_dict["unembed.W_U"] = classifier.weight.T
    # state_dict["unembed.b_U"] = classifier.bias

    # return state_dict
