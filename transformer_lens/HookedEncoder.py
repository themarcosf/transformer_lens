"""Hooked Encoder.

Contains a BERT style model. This is separate from :class:`transformer_lens.HookedTransformer`
because it has a significantly different architecture to e.g. GPT style transformers.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Union, cast, overload

import torch
from einops import repeat
from jaxtyping import Float, Int
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from transformers.activations import get_activation
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from typing_extensions import Literal

import transformer_lens.loading_from_pretrained as loading
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import (
    BertBlock,
    BertEmbed,
    BertMLMHead,
    BertNSPHead,
    BertPooler,
    Unembed,
)
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities import devices

import math


class Embeddings(nn.Module):
    def __init__(self, config: HookedTransformerConfig):
        super().__init__()
        self.cfg = config

        self.word_embeddings = nn.Embedding(self.cfg.d_vocab, self.cfg.d_model, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.cfg.n_ctx, self.cfg.d_model)

        self.LayerNorm = nn.LayerNorm(self.cfg.d_model, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer(
            "position_ids", torch.arange(self.cfg.n_ctx).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
            input_embeds (*optional*, torch.Tensor):
                The pre-computed word embeddings. Can only be passed if the input ids are `None`.


        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)

        seq_length = input_embeds.size(1)

        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: HookedTransformerConfig):
        super().__init__()
        self.cfg = config

        self.n_heads = self.cfg.n_heads
        self.dim = self.cfg.d_model
        self.dropout = nn.Dropout(p=0.1)
        self.is_causal = False

        # Have an even number of multi heads that divide the dimensions
        if self.dim % self.n_heads != 0:
            # Raise value errors for even multi-head attention nodes
            raise ValueError(f"self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly")

        self.q_lin = nn.Linear(in_features=self.cfg.d_model, out_features=self.cfg.d_model)
        self.k_lin = nn.Linear(in_features=self.cfg.d_model, out_features=self.cfg.d_model)
        self.v_lin = nn.Linear(in_features=self.cfg.d_model, out_features=self.cfg.d_model)
        self.out_lin = nn.Linear(in_features=self.cfg.d_model, out_features=self.cfg.d_model)

        self.pruned_heads = set()
        self.attention_head_size = self.dim // self.n_heads

    def prune_heads(self, heads: List[int]):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = self.attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)

class DistilBertSdpaAttention(MultiHeadSelfAttention):
    def __init__(self, config: HookedTransformerConfig):
        super().__init__(config=config)
        self.dropout_prob = 0.1
        self.require_contiguous_qkv = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        if output_attentions or head_mask is not None:
            print(
                "DistilBertSdpaAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support"
                " `output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but specifying"
                " the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be"
                ' removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                query,
                key,
                value,
                mask,
                head_mask,
                output_attentions,
            )

        batch_size, _, _ = query.size()
        dim_per_head = self.dim // self.n_heads

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and q.device.type == "cuda" and mask is not None:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )

        attn_output = unshape(attn_output)
        attn_output = self.out_lin(attn_output)

        return (attn_output,)
    
class FFN(nn.Module):
    def __init__(self, config: HookedTransformerConfig):
        super().__init__()

        self.cfg = config

        self.dropout = nn.Dropout(p=0.1)
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.lin1 = nn.Linear(in_features=self.cfg.d_model, out_features=self.cfg.d_mlp)
        self.lin2 = nn.Linear(in_features=self.cfg.d_mlp, out_features=self.cfg.d_model)
        self.activation = get_activation(self.cfg.act_fn)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input: torch.Tensor) -> torch.Tensor:
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: HookedTransformerConfig):
        super().__init__()

        self.cfg = config

        # Have an even number of Configure multi-heads
        if self.cfg.d_model % self.cfg.n_heads != 0:
            raise ValueError(f"self.cfg.n_heads {self.cfg.n_heads} must divide self.cfg.d_model {self.cfg.d_model} evenly")

        self.attention = DistilBertSdpaAttention(self.cfg)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=self.cfg.d_model, eps=1e-12)

        self.ffn = FFN(self.cfg)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=self.cfg.d_model, eps=1e-12)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
        ) -> Tuple[torch.Tensor, ...]:
            """
            Parameters:
                x: torch.tensor(bs, seq_length, dim)
                attn_mask: torch.tensor(bs, seq_length)

            Returns:
                sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
                torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
            """
            # Self-Attention
            sa_output = self.attention(
                query=x,
                key=x,
                value=x,
                mask=attn_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
            if output_attentions:
                sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
            else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
                if type(sa_output) is not tuple:
                    raise TypeError(f"sa_output must be a tuple but it is {type(sa_output)} type")

                sa_output = sa_output[0]
            sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

            # Feed Forward Network
            ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
            ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

            output = (ffn_output,)
            if output_attentions:
                output = (sa_weights,) + output
            return output

class Transformer(nn.Module):
    def __init__(self, config: HookedTransformerConfig):
        super().__init__()

        self.cfg = config

        self.n_layers = self.cfg.n_layers
        self.layer = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_state,
                attn_mask
            )
                
            hidden_state = layer_outputs[-1]

        return BaseModelOutput(last_hidden_state=hidden_state)
            
class DistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config: HookedTransformerConfig):
        model = DistilBertForSequenceClassification.from_pretrained(config.tokenizer_name)

        super().__init__(model.config)

        self.cfg = config

        self.embeddings = Embeddings(self.cfg)  # Embeddings
        self.transformer = Transformer(self.cfg)  # Encoder
        self._use_flash_attention_2 = False
        self._use_sdpa = True

        self.hook_embeddings = HookPoint()
        self.hook_transformer = HookPoint()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()

        embeddings = self.embeddings(input_ids)

        attention_mask = _prepare_4d_attention_mask_for_sdpa(
            attention_mask, embeddings.dtype, tgt_len=input_shape[1]
        )

        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
        )

class HookedEncoder(HookedRootModule):
    """
    This class implements a BERT-style encoder using the components in ./components.py, with HookPoints on every interesting activation. It inherits from HookedRootModule.

    Limitations:
    - The model does not include dropouts, which may lead to inconsistent results from training or fine-tuning.

    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`. There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
    """

    def __init__(self, cfg, tokenizer=None, move_to_device=True, **kwargs):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedEncoder.from_pretrained() instead."
            )
        self.cfg = cfg

        assert self.cfg.n_devices == 1, "Multiple devices not supported for HookedEncoder"
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif self.cfg.tokenizer_name is not None:
            huggingface_token = os.environ.get("HF_TOKEN", "")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_name,
                token=huggingface_token if len(huggingface_token) > 0 else None,
            )
        else:
            self.tokenizer = None

        if self.cfg.d_vocab == -1:
            # If we have a tokenizer, vocab size can be inferred from it.
            assert self.tokenizer is not None, "Must provide a tokenizer if d_vocab is not provided"
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab

        if self.cfg.original_architecture == 'DistilBertForSequenceClassification':
            self.distilbert = DistilBertModel(self.cfg)
            self.pre_classifier = nn.Linear(self.cfg.d_model, self.cfg.d_model)
            self.classifier = nn.Linear(self.cfg.d_model, 2)
            self.dropout = nn.Dropout(0.2)

            self.hook_distilbert = HookPoint()
            self.hook_pre_classifier = HookPoint()
            self.hook_classifier = HookPoint()
        else:
            self.embed = BertEmbed(self.cfg)
            self.blocks = nn.ModuleList([BertBlock(self.cfg) for _ in range(self.cfg.n_layers)])
            self.mlm_head = BertMLMHead(self.cfg)
            self.unembed = Unembed(self.cfg)
            self.nsp_head = BertNSPHead(self.cfg)
            self.pooler = BertPooler(self.cfg)

        self.hook_full_embed = HookPoint()

        if move_to_device:
            self.to(self.cfg.device)

        self.setup()

    def to_tokens(
        self,
        input: Union[str, List[str]],
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> Tuple[
        Int[torch.Tensor, "batch pos"],
        Int[torch.Tensor, "batch pos"],
        Int[torch.Tensor, "batch pos"],
    ]:
        """Converts a string to a tensor of tokens.
        Taken mostly from the HookedTransformer implementation, but does not support default padding
        sides or prepend_bos.
        Args:
            input (Union[str, List[str]]): The input to tokenize.
            move_to_device (bool): Whether to move the output tensor of tokens to the device the model lives on. Defaults to True
            truncate (bool): If the output tokens are too long, whether to truncate the output
            tokens to the model's max context window. Does nothing for shorter inputs. Defaults to
            True.
        """

        assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"

        encodings = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
        )

        tokens = encodings.input_ids

        if move_to_device:
            tokens = tokens.to(self.cfg.device)
            token_type_ids = encodings.token_type_ids.to(self.cfg.device)
            attention_mask = encodings.attention_mask.to(self.cfg.device)

        return tokens, token_type_ids, attention_mask

    def encoder_output(
        self,
        tokens: Int[torch.Tensor, "batch pos"],
        token_type_ids: Optional[Int[torch.Tensor, "batch pos"]] = None,
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_vocab"]:
        """Processes input through the encoder layers and returns the resulting residual stream.

        Args:
            input: Input tokens as integers with shape (batch, position)
            token_type_ids: Optional binary ids indicating segment membership.
                Shape (batch_size, sequence_length). For example, with input
                "[CLS] Sentence A [SEP] Sentence B [SEP]", token_type_ids would be
                [0, 0, ..., 0, 1, ..., 1, 1] where 0 marks tokens from sentence A
                and 1 marks tokens from sentence B.
            one_zero_attention_mask: Optional binary mask of shape (batch_size, sequence_length)
                where 1 indicates tokens to attend to and 0 indicates tokens to ignore.
                Used primarily for handling padding in batched inputs.

        Returns:
            resid: Final residual stream tensor of shape (batch, position, d_model)

        Raises:
            AssertionError: If using string input without a tokenizer
        """

        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(self.cfg.device)
            if one_zero_attention_mask is not None:
                one_zero_attention_mask = one_zero_attention_mask.to(self.cfg.device)

        resid = self.hook_full_embed(self.embed(tokens, token_type_ids))

        large_negative_number = -torch.inf
        mask = (
            repeat(1 - one_zero_attention_mask, "batch pos -> batch 1 1 pos")
            if one_zero_attention_mask is not None
            else None
        )
        additive_attention_mask = (
            torch.where(mask == 1, large_negative_number, 0) if mask is not None else None
        )

        for block in self.blocks:
            resid = block(resid, additive_attention_mask)

        return resid

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        return SequenceClassifierOutput(
            logits=logits
        )

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], ActivationCache,]:
        ...

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[False], **kwargs
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], Dict[str, torch.Tensor],]:
        ...

    def run_with_cache(
        self,
        *model_args,
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_vocab"],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True, this will return an ActivationCache object, with a bunch of useful HookedTransformer specific methods, otherwise it will return a dictionary of activations as in HookedRootModule. This function was copied directly from HookedTransformer.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict

    def to(  # type: ignore
        self,
        device_or_dtype: Union[torch.device, str, torch.dtype],
        print_details: bool = True,
    ):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cuda")

    def cpu(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cpu")

    def mps(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("mps")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model=None,
        device: Optional[str] = None,
        tokenizer=None,
        move_to_device=True,
        dtype=torch.float32,
        **from_pretrained_kwargs,
    ) -> HookedEncoder:
        """Loads in the pretrained weights from huggingface. Currently supports loading weight from HuggingFace BertForMaskedLM. Unlike HookedTransformer, this does not yet do any preprocessing on the model."""
        logging.warning(
            "Support for BERT in TransformerLens is currently experimental, until such a time when it has feature "
            "parity with HookedTransformer and has been tested on real research tasks. Until then, backward "
            "compatibility is not guaranteed. Please see the docs for information on the limitations of the current "
            "implementation."
            "\n"
            "If using BERT for interpretability research, keep in mind that BERT has some significant architectural "
            "differences to GPT. For example, LayerNorms are applied *after* the attention and MLP components, meaning "
            "that the last LayerNorm in a block cannot be folded."
        )

        assert not (
            from_pretrained_kwargs.get("load_in_8bit", False)
            or from_pretrained_kwargs.get("load_in_4bit", False)
        ), "Quantization not supported"

        if "torch_dtype" in from_pretrained_kwargs:
            dtype = from_pretrained_kwargs["torch_dtype"]

        official_model_name = loading.get_official_model_name(model_name)

        cfg = loading.get_pretrained_model_config(
            official_model_name,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=False,
            device=device,
            n_devices=1,
            dtype=dtype,
            **from_pretrained_kwargs,
        )

        state_dict = loading.get_pretrained_state_dict(
            official_model_name, cfg, hf_model, dtype=dtype, **from_pretrained_kwargs
        )

        model = cls(cfg, tokenizer, move_to_device=False)

        model.load_state_dict(state_dict, strict=False)

        if move_to_device:
            model.to(cfg.device)

        print(f"Loaded pretrained model {model_name} into HookedEncoder")

        return model

    @property
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        """
        Convenience to get the unembedding matrix (ie the linear map from the final residual stream to the output logits)
        """
        return self.unembed.W_U

    @property
    def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
        """
        Convenience to get the unembedding bias
        """
        return self.unembed.b_U

    @property
    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        """
        Convenience to get the embedding matrix
        """
        return self.embed.embed.W_E

    @property
    def W_pos(self) -> Float[torch.Tensor, "n_ctx d_model"]:
        """
        Convenience function to get the positional embedding. Only works on models with absolute positional embeddings!
        """
        return self.embed.pos_embed.W_pos

    @property
    def W_E_pos(self) -> Float[torch.Tensor, "d_vocab+n_ctx d_model"]:
        """
        Concatenated W_E and W_pos. Used as a full (overcomplete) basis of the input space, useful for full QK and full OV circuits.
        """
        return torch.cat([self.W_E, self.W_pos], dim=0)

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the key weights across all layers"""
        return torch.stack([cast(BertBlock, block).attn.W_K for block in self.blocks], dim=0)

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the query weights across all layers"""
        return torch.stack([cast(BertBlock, block).attn.W_Q for block in self.blocks], dim=0)

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the value weights across all layers"""
        return torch.stack([cast(BertBlock, block).attn.W_V for block in self.blocks], dim=0)

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stacks the attn output weights across all layers"""
        return torch.stack([cast(BertBlock, block).attn.W_O for block in self.blocks], dim=0)

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stacks the MLP input weights across all layers"""
        return torch.stack([cast(BertBlock, block).mlp.W_in for block in self.blocks], dim=0)

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stacks the MLP output weights across all layers"""
        return torch.stack([cast(BertBlock, block).mlp.W_out for block in self.blocks], dim=0)

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the key biases across all layers"""
        return torch.stack([cast(BertBlock, block).attn.b_K for block in self.blocks], dim=0)

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the query biases across all layers"""
        return torch.stack([cast(BertBlock, block).attn.b_Q for block in self.blocks], dim=0)

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the value biases across all layers"""
        return torch.stack([cast(BertBlock, block).attn.b_V for block in self.blocks], dim=0)

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the attn output biases across all layers"""
        return torch.stack([cast(BertBlock, block).attn.b_O for block in self.blocks], dim=0)

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stacks the MLP input biases across all layers"""
        return torch.stack([cast(BertBlock, block).mlp.b_in for block in self.blocks], dim=0)

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the MLP output biases across all layers"""
        return torch.stack([cast(BertBlock, block).mlp.b_out for block in self.blocks], dim=0)

    @property
    def QK(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the Q and K matrices for each layer and head.
        Useful for visualizing attention patterns."""
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the O and V matrices for each layer and head."""
        return FactoredMatrix(self.W_V, self.W_O)

    def all_head_labels(self) -> List[str]:
        """Returns a list of strings with the format "L{l}H{h}", where l is the layer index and h is the head index."""
        return [f"L{l}H{h}" for l in range(self.cfg.n_layers) for h in range(self.cfg.n_heads)]
    
