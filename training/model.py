import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value, and o projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 4 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.vis_n_embd, 4 * config.vis_n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(4 * config.vis_n_embd, config.vis_n_embd)
        self.fc2.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class LLMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(14336, config.llm_n_embd)
        self.up_proj = nn.Linear(14336, config.llm_n_embd)
        self.down_proj = nn.Linear(config.llm_n_embd, 14336)

    def forward(self, x):
        x = self.gate_proj(x)
        x = self.up_proj(x)
        x = self.down_proj(x)
        return x


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(config.vis_n_embd, config.vis_n_embd),
                "k_proj": nn.Linear(config.vis_n_embd, config.vis_n_embd),
                "v_proj": nn.Linear(config.vis_n_embd, config.vis_n_embd),
                "o_proj": nn.Linear(config.vis_n_embd, config.vis_n_embd),
            }
        )
        self.mlp = VisionMLP(config)
        self.input_layernorm = nn.LayerNorm(config.vis_n_embd)
        self.post_attention_layernorm = nn.LayerNorm(config.vis_n_embd)

    def forward(self, x):
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LLMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(config.llm_n_embd, config.llm_n_embd),
                "k_proj": nn.Linear(config.llm_n_embd // 4, config.llm_n_embd),
                "v_proj": nn.Linear(config.llm_n_embd // 4, config.llm_n_embd),
                "o_proj": nn.Linear(config.llm_n_embd, config.llm_n_embd),
            }
        )
        self.mlp = LLMMLP(config)
        self.input_layernorm = nn.LayerNorm(config.llm_n_embd)
        self.post_attention_layernorm = nn.LayerNorm(config.llm_n_embd)

    def forward(self, x):
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


@dataclass
class Llama_32_11B_VisionConfig:
    img_size: int = 1280  # image size
    vis_n_embd: int = 1280  # vision embedding dimension
    vis_transformer_n_layers: int = 32  # number of layers in the vision transformer
    vis_global_transformer_n_layers: int = 8  # number of layers in the global transformer

    block_size: int = 1024  # max sequence length
    n_head: int = 12  # number of heads

    vocab_size: int = 128264  # number of tokens: 128,007 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    llm_n_embd: int = 4096  # llm embedding dimension
    llm_n_layers: int = 40  # number of layers in the llm


class Llama_32_11B_Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vision_model = nn.ModuleDict(
            {
                "class_embedding": nn.Embedding(config.img_size, 1),
                "patch_embedding": nn.Embedding(config.img_size, 3 * 14 * 14),
                "gated_positional_embedding": nn.ModuleDict(
                    {
                        "gate": nn.Linear(1, 1, bias=False),
                        "embedding": nn.Embedding(1601, config.vis_n_embd),
                        "tile_embedding": nn.Embedding(9, 8197120),
                    }
                ),
                "pre_tile_positional_embedding": nn.ModuleDict(
                    {
                        "gate": nn.Linear(1, 1, bias=False),
                        "embedding": nn.Embedding(9, 4 * config.vis_n_embd),
                    }
                ),
                "post_tile_positional_embedding": nn.ModuleDict(
                    {
                        "gate": nn.Linear(1, 1, bias=False),
                        "embedding": nn.Embedding(9, 4 * config.vis_n_embd),
                    }
                ),
                "layernorm_pre": nn.LayerNorm(config.vis_n_embd),
                "layernorm_post": nn.LayerNorm(config.vis_n_embd),
                "transformer": nn.ModuleList([VisionBlock(config) for _ in range(config.vis_transformer_n_layers)]),
                "global_transformer": nn.ModuleDict(
                    {
                        "layers": nn.ModuleList(
                            [
                                nn.ModuleDict(
                                    {
                                        "gate_attn": nn.Linear(1, 1, bias=False),
                                        "gate_ffn": nn.Linear(1, 1, bias=False),
                                        "self_attn": nn.ModuleDict(
                                            {
                                                "q_proj": nn.Linear(config.vis_n_embd, config.vis_n_embd),
                                                "k_proj": nn.Linear(config.vis_n_embd, config.vis_n_embd),
                                                "v_proj": nn.Linear(config.vis_n_embd, config.vis_n_embd),
                                                "o_proj": nn.Linear(config.vis_n_embd, config.vis_n_embd),
                                            }
                                        ),
                                        "mlp": nn.ModuleDict(
                                            {
                                                "fc1": nn.Linear(4 * config.vis_n_embd, config.vis_n_embd),
                                                "fc2": nn.Linear(config.vis_n_embd, 4 * config.vis_n_embd),
                                            }
                                        ),
                                        "input_layernorm": nn.LayerNorm(config.vis_n_embd),
                                        "post_attention_layernorm": nn.LayerNorm(config.vis_n_embd),
                                    }
                                )
                                for _ in range(config.vis_global_transformer_n_layers)
                            ]
                        )
                    }
                ),
            }
        )

        self.language_model = nn.ModuleDict(
            {
                "embed_tokens": nn.Embedding(config.vocab_size, config.llm_n_embd),
                "layers": nn.ModuleList([LLMBlock(config) for _ in range(config.n_layer)]),
                "norm": nn.LayerNorm(config.llm_n_embd),
                "lm_head": nn.Linear(config.vocab_size, config.llm_n_embd, bias=False),
            }
        )

        self.multi_modal_projector = nn.Linear(7680, config.llm_n_embd)

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls):
        """Loads pretrained Llama-3.2-11B-Vision model weights from huggingface"""
        from transformers import MllamaForConditionalGeneration

        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        print(f"loading weights from pretrained model: {model_id}")

        # create a from-scratch initialized model
        config = Llama_32_11B_VisionConfig()
        model = Llama_32_11B_Vision(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        # init a huggingface/transformers model
        model_hf = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = dict(self.named_parameters())
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
