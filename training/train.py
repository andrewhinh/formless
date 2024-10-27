# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# import logging
# import sys
# import time
# from functools import partial
# from importlib import import_module
# from types import ModuleType
# from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union
# from warnings import warn

# import torch
# import torch.nn.functional as F
# from omegaconf import DictConfig, ListConfig
# from torch import nn
# from torch.optim import Optimizer
# from torch.optim.lr_scheduler import LRScheduler
# from torch.utils.data import DataLoader, Dataset, DistributedSampler
# from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION
# from torchtune.utils._logging import log_once
# from tqdm import tqdm

# log = utils.get_logger("DEBUG")


# class InstantiationError(Exception):
#     """
#     Raised when a `_component_` field in a config is unable to be instantiated.
#     """

#     pass


# CROSS_ENTROPY_IGNORE_IDX = -100
# PACK_TYPE = Dict[str, Union[torch.Tensor, List[int]]]


# PROFILER_KEY = "profiler"


# if _SUPPORTS_FLEX_ATTENTION:
#     from torch.nn.attention.flex_attention import (
#         BlockMask,
#         flex_attention,
#     )
#     from torch.nn.attention.flex_attention import (
#         create_block_mask as create_block_causal_mask_flex,
#     )

#     flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

#     # We cannot do nested compile, but flex attention only has perf benefits
#     # when compiled. To insulate it from the compiler, we wrap it with
#     # compiler.disable so that it can be used regardless of whether the model
#     # is compiled or not, and flex attention always remains compiled.
#     @torch.compiler.disable(recursive=False)
#     def compile_friendly_flex_attention(
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         block_mask: BlockMask,
#     ) -> torch.Tensor:
#         return flex_attention_compiled(q, k, v, block_mask=block_mask)

#     _MaskType = Union[torch.Tensor, BlockMask]
# else:
#     _MaskType = torch.Tensor


# def _get_document_ids_from_seq_lens(
#     seq_lens: List[torch.Tensor],
# ) -> torch.Tensor:
#     """
#     Convert a batch tensor of seq lens into integer IDs denoting sample ownership.
#     For example, seq_lens = [2, 3, 1] would return [0, 0, 1, 1, 1, 2].

#     Args:
#         seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
#             shape (batch_size, n), where n is the max number of sequences in a pack and can vary
#             across packs.

#     Returns
#     -------
#         Tensor: Document IDs of shape (batch_size, max_seq_len).
#     """
#     batch_size = len(seq_lens)
#     batch_document_ids = []
#     for sample_idx in range(batch_size):
#         # We assume seq lens sum to max seq lens, so document_ids should be of
#         # shape (max_seq_len, )
#         document_ids = torch.cat(
#             [
#                 torch.full((seq_len,), i, dtype=torch.long, device=seq_len.device)
#                 for i, seq_len in enumerate(seq_lens[sample_idx])
#             ]
#         )
#         batch_document_ids.append(document_ids)
#     batch_document_ids = torch.stack(batch_document_ids)
#     return batch_document_ids


# def create_block_causal_mask(seq_lens: List[torch.Tensor]) -> torch.Tensor:
#     """
#     Given a batch tensor of seq lens defining the lengths of samples in each pack,
#     Construct a 2D block causal mask for each pack in the batch. For example, if
#     a single sample's seq_lens is [3, 2, 1], the mask would be::

#         mask = [
#             [1, 0, 0, 0, 0, 0],
#             [1, 1, 0, 0, 0, 0],
#             [1, 1, 1, 0, 0, 0],
#             [0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 1, 1, 0],
#             [0, 0, 0, 0, 0, 1],
#         ]

#     Args:
#         seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
#             shape (batch_size, n), where n is the max number of sequences in a pack and can vary
#             across packs.


#     Returns
#     -------
#         Tensor: Block causal mask of shape (batch_size, max_seq_len, max_seq_len).
#     """
#     batch_block_attn_masks = []
#     batch_size = len(seq_lens)
#     for sample_idx in range(batch_size):
#         block_attn_masks = [
#             torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=seq_len.device))
#             for i, seq_len in enumerate(seq_lens[sample_idx])
#         ]

#         batch_block_attn_masks.append(torch.block_diag(*block_attn_masks))
#     return torch.stack(batch_block_attn_masks)


# def packed_block_causal_mask(
#     seq_lens: List[torch.Tensor],
# ) -> _MaskType:
#     """
#     Create a block causal document mask for a batch of packed sequences. If on
#     torch version >= 2.5.0, this is done by creating a mask_mod function with the
#     block causal logic and passing this into :func:`torch.nn.attention.flex_attention.create_block_mask`.
#     The resultant BlockMask is a compressed representation of the full block causal
#     mask. If on an older version, a standard 2D block causal mask is created and returned.

#     Args:
#         seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
#             shape (batch_size, n), where n is the max number of sequences in a pack and can vary
#             across packs.

#     Returns
#     -------
#         _MaskType: BlockMask or Tensor if torch version < 2.5.0.
#     """
#     if _SUPPORTS_FLEX_ATTENTION:
#         document_ids = _get_document_ids_from_seq_lens(seq_lens)
#         batch_size, max_seq_len = document_ids.shape
#         document_ids = document_ids.to("cuda")

#         # Instead of passing a tensor mask, flex attention requires a mask_mod function
#         # that determines which elements of QK^T should be included in the attention
#         # computation prior to the softmax. For sample packing, we need both the
#         # logic for both causal mask and document mask. See PyTorch's official
#         # blog post for more details: https://pytorch.org/blog/flexattention/#mask-mods
#         def mask_mod(b, h, q_idx, kv_idx):
#             """
#             Defines the logic of a block causal mask by combining both a standard causal mask
#             and a block diagonal document mask.

#             See :func:`~torchtune.modules.attention_utils.create_block_causal_mask`
#             for an illustration.
#             """
#             causal_mask = q_idx >= kv_idx
#             document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
#             return causal_mask & document_mask

#         return create_block_causal_mask_flex(
#             mask_mod,
#             batch_size,
#             None,
#             max_seq_len,
#             max_seq_len,
#             device="cuda",
#         )
#     else:
#         return create_block_causal_mask(seq_lens=seq_lens)


# def _sdpa_or_flex_attention() -> Callable:
#     """
#     Helper function to decide when to call flex attention or SDPA. It will use
#     flex attention if ALL of the following conditions are met, otherwise it will
#     default to SDPA:
#     - torch version >= 2.5.0
#     - we are sample packing, therefore mask is a BlockMask
#     - torch.cuda.get_device_capability() >= (7, 5)
#     """
#     if _SUPPORTS_FLEX_ATTENTION:

#         def _attention_call(
#             q: torch.Tensor,
#             k: torch.Tensor,
#             v: torch.Tensor,
#             mask: Optional[_MaskType],
#             dropout_p: float,
#             is_causal: bool,
#         ) -> torch.Tensor:
#             # Flex attention uses the BlockMask
#             # (https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py#L168)
#             # instead of a traditional boolean tensor mask. If this is passed in,
#             # we assume the user wants to use flex attention instead of traditional SDPA.
#             # This will use flash attention under the hood with support for custom masks.
#             # Currently, it is used when sample packing is enabled (see torchtune.datasets.PackedDataset)
#             if isinstance(mask, BlockMask):
#                 log_once(
#                     log,
#                     "Using flex attention for attention computation since a BlockMask was passed in.",
#                     level=logging.DEBUG,
#                 )
#                 if dropout_p > 0.0:
#                     raise ValueError("Flex attention does not support dropout. Please set dropout to 0.0.")
#                 return compile_friendly_flex_attention(
#                     q,
#                     k,
#                     v,
#                     block_mask=mask,
#                 )
#             # If mask is a standard boolean tensor or None, then use SDPA
#             else:
#                 # shape: [b, 1, s, s]
#                 if mask is not None:
#                     mask = mask[:, None, :, :]

#                 # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
#                 return nn.functional.scaled_dot_product_attention(
#                     q,
#                     k,
#                     v,
#                     attn_mask=mask,
#                     dropout_p=dropout_p,
#                     is_causal=is_causal,
#                 )

#     else:

#         def _attention_call(
#             q: torch.Tensor,
#             k: torch.Tensor,
#             v: torch.Tensor,
#             mask: Optional[_MaskType],
#             dropout_p: float,
#             is_causal: bool,
#         ) -> torch.Tensor:
#             # shape: [b, 1, s, s]
#             if mask is not None:
#                 mask = mask[:, None, :, :]

#             # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
#             return nn.functional.scaled_dot_product_attention(
#                 q,
#                 k,
#                 v,
#                 attn_mask=mask,
#                 dropout_p=dropout_p,
#                 is_causal=is_causal,
#             )

#     return _attention_call


# class DummyProfiler:
#     """
#     Drop-in replacement for torch.profiler.profile that functions as a nullcontext / object
#     with no-op methods for ``start``, ``stop``, and ``step``.

#     This is helpful for instrumenting profiling in a recipe without requiring changes to the
#     code independent of whether profiling is on / off.

#     E.g.,
#     ```
#         profiler = DummyProfiler()
#         #profiler = torch.profiler.profile()

#         # Below is same regardless of profiler object type
#         with profiler as prof:
#             for epoch in epochs:
#                 for batch in batches:
#                     train.step()
#                     prof.step()

#     """

#     def __enter__(self):
#         return self

#     def __exit__(self, *args):
#         pass

#     def start(self):
#         pass

#     def stop(self):
#         pass

#     def step(self):
#         pass


# class OptimizerInBackwardWrapper:
#     """
#     A bare-bones class meant for checkpoint save and load for optimizers running
#     in backward. Usage is limited to the following:

#     Note:
#         This wrapper is only meant to be used for single-device use cases.
#         Distributed use cases such as FSDP, which require specialized optimizer state checkpointing, are not supported.

#     Args:
#         optim_map (Dict[str, torch.optim.Optimizer]): Mapping from parameter names to optimizers.

#     Example:
#         >>> optim_dict = {
#         >>>     p: config.instantiate(cfg_optimizer, [p])
#         >>>     for p in self._model.parameters()
#         >>> }
#         >>>
#         >>> # Save checkpoint
#         >>> ckpt = OptimizerInBackwardWrapper(optim_dict).state_dict()
#         >>> torch.save("/tmp/optim_ckpt", ckpt)
#         >>>
#         >>> # Load checkpoint
#         >>> placeholder_optim_dict = {
#         >>>     p: config.instantiate(cfg_optimizer, [p])
#         >>>     for p in self._model.parameters()
#         >>> }
#         >>>
#         >>> wrapper = OptimInBackwardWrapper(placeholder_optim_dict)
#         >>>
#         >>> # load_state_dict expects a dict produced by this class's
#         >>> # state_dict method.
#         >>> wrapper.load_state_dict(torch.load("/tmp/optim_ckpt"))
#         >>> # placeholder_optim_dict now has updated optimizer states.

#     """

#     def __init__(self, optim_map: Dict[str, torch.optim.Optimizer]):
#         self.optim_map = optim_map
#         self.lr_scheduler = None

#     def state_dict(self) -> Dict[str, Any]:
#         """
#         Returns a state dict mapping parameter names to optimizer states. This
#         state_dict is only loadable by this same class.

#         Returns
#         -------
#             Dict[str, Any]: state dict mapping parameter names to optimizer states.
#         """
#         return {p: opt.state_dict() for p, opt in self.optim_map.items()}

#     def load_state_dict(self, optim_ckpt_map: Dict[str, Any]):
#         """
#         Load optimizer states from a state dict produced by this class's
#         state_dict method.

#         Args:
#             optim_ckpt_map (Dict[str, Any]): state dict mapping parameter names to optimizer states.

#         Raises
#         ------
#             RuntimeError: If the optimizer state dict does not contain all the expected parameters.
#         """
#         params_covered = set()
#         for param_name in optim_ckpt_map.keys():
#             if param_name not in self.optim_map:
#                 raise RuntimeError(f"Trying to load optimizer state for unexpected param {param_name}")
#             self.optim_map[param_name].load_state_dict(optim_ckpt_map[param_name])
#             params_covered.add(param_name)
#         # Ensure all params have been loaded into, report missing params
#         missing_params = set(self.optim_map.keys()) - params_covered
#         if missing_params:
#             raise RuntimeError(f"Expected to load optimizer state for params {missing_params}!")

#     def get_optim_key(self, key: str) -> Any:
#         """
#         Returns value of key from an arbitrary optimizer running in backward. Note that
#         this assumes all optimizer in backwards have the same value for the key, i.e.,
#         are initialized with the same hyperparameters.
#         """
#         return list(self.optim_map.values())[0].param_groups[0][key]

#     def set_lr_scheduler(self, lr_scheduler: LRScheduler) -> None:
#         """
#         Sets the learning rate scheduler and modifies its step method to update all optimizers.

#         Args:
#             lr_scheduler (LRScheduler): The learning rate scheduler to use.
#         """
#         self.lr_scheduler = lr_scheduler
#         original_step = self.lr_scheduler.step

#         def custom_step(epoch=None):
#             if epoch is None:
#                 original_step()
#             else:
#                 original_step(epoch)
#             new_lr = self.lr_scheduler.get_last_lr()[0]
#             for opt in self.optim_map.values():
#                 for param_group in opt.param_groups:
#                     param_group["lr"] = new_lr

#         self.lr_scheduler.step = custom_step

#     def step_lr_scheduler(self, epoch: int = None):
#         """
#         Steps the learning rate scheduler if it exists.

#         Args:
#             epoch (int, optional): The current epoch number. Defaults to None.

#         Raises
#         ------
#             RuntimeError: If the LR scheduler has not been set.
#         """
#         if self.lr_scheduler:
#             self.lr_scheduler.step(epoch)
#         else:
#             raise RuntimeError("LR scheduler has not been set. Call set_lr_scheduler first.")

#     def get_last_lr(self) -> float:
#         """
#         Gets the last learning rate from the scheduler if it exists.

#         Returns
#         -------
#             float: The last learning rate.

#         Raises
#         ------
#             RuntimeError: If the LR scheduler has not been set.
#         """
#         if self.lr_scheduler:
#             return self.lr_scheduler.get_last_lr()[0]
#         else:
#             raise RuntimeError("LR scheduler has not been set. Call set_lr_scheduler first.")


# def _get_component_from_path(path: str) -> Any:
#     """
#     Return an object by name or dotted path, importing as necessary.
#     The base functionality relies on ``getattr()`` and handles all
#     possible exceptions accordingly.

#     Based on Hydra's `_locate` from Facebook Research:
#     https://github.com/facebookresearch/hydra/blob/main/hydra/_internal/utils.py#L614

#     Args:
#         path (str): Dotted path of the object

#     Returns
#     -------
#         Any: The object

#     Raises
#     ------
#         InstantiationError: If there is an exception loading the
#             object from the provided path
#         ValueError: If a relative or invalid dotpath is passed in
#     """
#     if path == "":
#         raise ValueError("Empty path")

#     parts = [part for part in path.split(".")]
#     for part in parts:
#         # If a relative path is passed in, the first part will be empty
#         if not len(part):
#             raise ValueError(f"Error loading '{path}': invalid dotstring." + "\nRelative imports are not supported.")
#     # First module requires trying to import to validate
#     part0 = parts[0]
#     try:
#         obj = import_module(part0)
#     except ImportError as exc_import:
#         raise InstantiationError(
#             f"Error loading '{path}':\n{repr(exc_import)}" + f"\nAre you sure that module '{part0}' is installed?"
#         ) from exc_import
#     # Subsequent components can be checked via getattr() on first module
#     # It can either be an attribute that we can return or a submodule that we
#     # can import and continue searching
#     for m in range(1, len(parts)):
#         part = parts[m]
#         try:
#             obj = getattr(obj, part)
#         # If getattr fails, check to see if it's a module we can import and
#         # continue down the path
#         except AttributeError as exc_attr:
#             parent_dotpath = ".".join(parts[:m])
#             if isinstance(obj, ModuleType):
#                 mod = ".".join(parts[: m + 1])
#                 try:
#                     obj = import_module(mod)
#                     continue
#                 except ModuleNotFoundError as exc_import:
#                     raise InstantiationError(
#                         f"Error loading '{path}':\n{repr(exc_import)}"
#                         + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
#                     ) from exc_import
#                 # Any other error trying to import module can be raised as
#                 # InstantiationError
#                 except Exception as exc_import:
#                     raise InstantiationError(f"Error loading '{path}':\n{repr(exc_import)}") from exc_import
#             # If the component is not an attribute nor a module, it doesn't exist
#             raise InstantiationError(
#                 f"Error loading '{path}':\n{repr(exc_attr)}"
#                 + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
#             ) from exc_attr
#     return obj


# def padded_collate_packed(
#     batch: List[PACK_TYPE],
# ) -> Dict[str, torch.Tensor]:
#     """Collate packed sequences into a batch. Only convert the seq lens into
#     a block mask for use with attention. Tokens, labels, and input_pos are
#     already padded to the same length within :class:`~torchtune.datasets.PackedDataset`.

#     Args:
#         batch (List[PACK_TYPE]): A list of pack dictionaries containing the following keys:
#             - tokens: input token ids
#             - labels: label token ids
#             - input_pos: relative position ids for each sequence in pack
#             - seq_lens: lengths of each sample within the pack

#     Returns
#     -------
#         Dict[str, torch.Tensor]: Collated input, label, input_pos, mask tensors.

#     Example:
#         >>> token_pairs = [
#         >>>    {"tokens": [1, 2, 3, 4, 5, 6], "labels": [7, 8, 9, 10, 11, 12],
#         >>>     "input_pos": [0, 1, 2, 0, 1, 0], "seq_lens": [3, 2, 1]},
#         >>>    {"tokens": [13, 14, 15, 16, 17, 18], "labels": [19, 20, 21, 22, 23, 24],
#         >>>     "input_pos": [0, 1, 0, 1, 0, 1], "seq_lens": [2, 2, 2]},
#         >>> ]
#         >>> collated = padded_collate_packed(
#         >>>    batch=token_pairs,
#         >>>    device=device,
#         >>> )
#         >>> collated["mask"]
#         >>> tensor([
#         >>> [[1, 0, 0, 0, 0, 0],
#         >>>  [1, 1, 0, 0, 0, 0],
#         >>>  [1, 1, 1, 0, 0, 0],
#         >>>  [0, 0, 0, 1, 0, 0],
#         >>>  [0, 0, 0, 1, 1, 0],
#         >>>  [0, 0, 0, 0, 0, 1]],
#         >>> [[1, 0, 0, 0, 0, 0],
#         >>>  [1, 1, 0, 0, 0, 0],
#         >>>  [0, 0, 1, 0, 0, 0],
#         >>>  [0, 0, 1, 1, 0, 0],
#         >>>  [0, 0, 0, 0, 1, 0],
#         >>>  [0, 0, 0, 0, 1, 1]])
#     """
#     tokens = torch.stack([x["tokens"] for x in batch])
#     labels = torch.stack([x["labels"] for x in batch])
#     input_pos = torch.stack([x["input_pos"] for x in batch])
#     seq_lens = [x["seq_lens"] for x in batch]

#     block_mask = packed_block_causal_mask(
#         seq_lens=seq_lens,
#     )

#     return {
#         "tokens": tokens,
#         "labels": labels,
#         "input_pos": input_pos,
#         "mask": block_mask,
#     }


# class PackedDataset(Dataset):
#     """
#     Performs greedy sample packing on a provided dataset. This is done as a single
#     preprocessing step before training begins. Shuffling is done outside of this
#     class on packed samples with a ``Sampler`` as part of the dataloader. Currently,
#     this only supports in-memory map-style datasets.

#     The class loads, tokenizes, and packs examples on initialization - no tokenization is done during training.

#     The general flow on initialization is: load tokenized sample -> add to buffer ->
#     when buffer is long enough, add to ``self.packs``.

#     During training, returns self.packs[idx] as input, label, attention mask, and
#     position ids. The attention mask is a lower triangular block mask to prevent
#     samples from cross-attending within a pack. The position ids indicate the position
#     of each token relative to its sample within a pack. These are all padded to max
#     sequence length, so a batch-wise collator is not needed.

#     A packed sample is made up of individual smaller sequence length samples jammed together
#     within ``max_seq_len``. For example, if max_seq_len is 6 and there are varied
#     length samples::

#         tokens = [
#             [S1, S1, S1, S2, S2, pad],
#             [S3, S3, S4, S4, pad, pad],
#             ...,
#         ]

#     To prevent cross-contamination, the following mask would be returned for the
#     first pack in the example::

#         mask = [
#             [1, 0, 0, 0, 0, 0],
#             [1, 1, 0, 0, 0, 0],
#             [1, 1, 1, 0, 0, 0],
#             [0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 1, 1, 0],
#             [0, 0, 0, 0, 0, 1],
#         ]

#     The position ids would be::

#         input_pos = [
#             [0, 1, 2, 0, 1, 2],
#             [0, 1, 0, 1, 2, 3],
#             ...,
#         ]

#     The identity matrix is used in the mask for pad tokens instead of a causal mask.
#     For position ids for pad tokens, we simply continue to increment from the previous
#     sample normally.

#     Args:
#         ds (Dataset): dataset to sample pack. This should return a dictionary with field
#             "tokens" and "labels" containing the tokenized and label samples.
#         max_seq_len (int): Maximum number of tokens to pack
#         padding_idx (int): padding index for the tokenizer. Default is 0.
#         max_packs (Optional[int]): Maximum number of packs. Default is None, which will create as many
#             packs as possible.
#         split_across_pack (bool): if the last sample in a pack does not fit in ``max_seq_len``,
#             split the sample into the next pack, or move it entirely to the beginning of the next pack.
#             For pre-training, typically this is set to True for general text completion. For
#             fine-tuning, typically this is set to False to avoid truncating sentences in instruct
#             tuning. Default is False.
#     """

#     def __init__(
#         self,
#         ds: Dataset,
#         *,
#         max_seq_len: int,
#         padding_idx: int = 0,
#         max_packs: Optional[int] = None,
#         split_across_pack: bool = False,
#     ) -> None:
#         self.ds = ds
#         self.max_seq_len = max_seq_len
#         self.padding_idx = padding_idx
#         self.max_packs = max_packs
#         self.split_across_pack = split_across_pack
#         # Where final samples will be held
#         self.packs: List[PACK_TYPE] = []
#         self.previous_sample_boundary: int = 0
#         self._pack()

#     def _pack(self) -> None:
#         """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
#         then append the buffer to self.packs as a single "packed" sample. Continue
#         until max_packs or end of dataset.
#         """
#         # Buffer to hold samples until they are long enough to be added to self.packs
#         current_pack = {
#             "tokens": [],
#             "labels": [],
#             "input_pos": [],
#             "seq_lens": [],
#         }

#         # Only show progress bar on rank 0
#         rank = (
#             torch.distributed.get_rank()
#             if torch.distributed.is_available() and torch.distributed.is_initialized()
#             else 0
#         )
#         if rank == 0:
#             pbar = tqdm(total=len(self.ds), desc="Packing dataset", dynamic_ncols=True)

#         for sample in self.ds:
#             tokens, labels = sample["tokens"], sample["labels"]

#             # If the dataset outputs samples that are larger than the specified
#             # max_seq_len and we're unable to split it, user needs to modify
#             # one of the two parameters
#             seq_len = len(tokens)
#             if seq_len > self.max_seq_len and not self.split_across_pack:
#                 raise ValueError(
#                     f"Dataset sample is too long ({seq_len} > {self.max_seq_len}). "
#                     "Please set `split_across_pack=True` or increase `max_seq_len`."
#                 )

#             # Update the current pack
#             current_pack["tokens"] += tokens
#             current_pack["labels"] += labels
#             current_pack["input_pos"] += [x % self.max_seq_len for x in range(seq_len)]
#             current_pack["seq_lens"] += [seq_len]

#             # If the current pack is over the max_seq_len, add it to self.packs and
#             # retain any truncated or bumped samples for next pack
#             while len(current_pack["tokens"]) > self.max_seq_len and not self._should_stop_packing():
#                 current_pack = self._split_and_add_pack(current_pack)

#             if rank == 0:
#                 pbar.update()

#             # Keep track of previous sample boundary
#             self.previous_sample_boundary = len(current_pack["tokens"])

#             if self._should_stop_packing():
#                 break

#         # Handle the last pack if there's leftover and we haven't filled up the max packs
#         if len(current_pack["tokens"]) > 0 and (self.max_packs is None or len(self.packs) < self.max_packs):
#             # No need to handle splitting at this point so we can just add the current pack
#             self._add_pack(current_pack)

#     def _should_stop_packing(self) -> bool:
#         """If max packs is set, stop packing when we reach that number."""
#         if self.max_packs is not None and len(self.packs) == self.max_packs:
#             return True
#         return False

#     def _split_and_add_pack(self, current_pack: PACK_TYPE) -> PACK_TYPE:
#         """Splits the current pack at the boundary, processes it, adds it to ``self.packs`` and
#         returns the start of the next pack.
#         """
#         if self.split_across_pack:
#             boundary = self.max_seq_len
#             # The last elem in ``seq_lens`` ensures that ``sum(seq_lens) == self.max_seq_len``
#             leftover_seq_len = self.max_seq_len - sum(current_pack["seq_lens"][:-1])
#             seq_len_padding = [leftover_seq_len] if leftover_seq_len > 0 else []
#         else:
#             boundary = self.previous_sample_boundary
#             # If we aren't splitting across packs, we leave out the last sample b/c
#             # it will go into the next pack
#             seq_len_padding = []

#         pack = {
#             "tokens": current_pack["tokens"][:boundary],
#             "labels": current_pack["labels"][:boundary],
#             "input_pos": current_pack["input_pos"][:boundary],
#             "seq_lens": current_pack["seq_lens"][:-1] + seq_len_padding,
#         }

#         # Process and add the pack
#         self._add_pack(pack)

#         # Return the length of the first sample in next pack if we are splitting across packs,
#         # otherwise return the length of the last sample in the current pack
#         next_seq_len = (
#             len(current_pack["tokens"][boundary:]) if self.split_across_pack else current_pack["seq_lens"][-1]
#         )

#         return {
#             "tokens": current_pack["tokens"][boundary:],
#             "labels": current_pack["labels"][boundary:],
#             "input_pos": current_pack["input_pos"][boundary:],
#             "seq_lens": [next_seq_len],
#         }

#     def _add_pack(self, pack: PACK_TYPE) -> None:
#         """Processes, pads and adds a pack to ``self.packs``."""
#         pack = self._convert_to_tensors(pack)
#         pack = self._pad_pack(pack, padding_idx=self.padding_idx)
#         self.packs.append(pack)

#     def _convert_to_tensors(self, pack: PACK_TYPE) -> PACK_TYPE:
#         """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors."""
#         return {
#             "tokens": torch.tensor(pack["tokens"], dtype=torch.long),
#             "labels": torch.tensor(pack["labels"], dtype=torch.long),
#             "input_pos": torch.tensor(pack["input_pos"], dtype=torch.long),
#             "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long),
#         }

#     def _pad_pack(self, pack: PACK_TYPE, padding_idx: int) -> PACK_TYPE:
#         """Pads a pack to ``self.max_seq_len``."""
#         # Pad tokens
#         num_padding_tokens = self.max_seq_len - len(pack["tokens"])
#         padded_tokens = F.pad(
#             pack["tokens"],
#             (0, num_padding_tokens),
#             value=padding_idx,
#         )

#         # Pad labels
#         padded_labels = F.pad(
#             pack["labels"],
#             (0, self.max_seq_len - len(pack["labels"])),
#             value=CROSS_ENTROPY_IGNORE_IDX,
#         )

#         # Add padding tokens as a last seq len to ensure sum is max_seq_len
#         padded_seq_lens = (
#             torch.cat([pack["seq_lens"], torch.tensor([num_padding_tokens])])
#             if num_padding_tokens > 0
#             else pack["seq_lens"]
#         )

#         # Pad input_pos continuing the sequence from last value
#         # in input_pos
#         # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.max_seq_len = 6
#         num_range = torch.arange(
#             pack["input_pos"][-1] + 1,
#             pack["input_pos"][-1] + self.max_seq_len - len(pack["input_pos"]) + 1,
#         )
#         # Clamp to max_seq_len - 1 to avoid out of bounds error
#         clamped_num_range = torch.clamp(num_range, 0, self.max_seq_len - 1)
#         padded_input_pos = torch.cat([pack["input_pos"], clamped_num_range])

#         return {
#             "tokens": padded_tokens,
#             "labels": padded_labels,
#             "input_pos": padded_input_pos,
#             "seq_lens": padded_seq_lens,
#         }

#     def __len__(self) -> int:
#         return len(self.packs)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         return self.packs[idx]


# class ConcatDataset(Dataset):
#     """
#     A dataset class for concatenating multiple sub-datasets into a single dataset. This class enables the
#     unified handling of different datasets as if they were a single dataset, simplifying tasks such as
#     training models on multiple sources of data simultaneously.

#     The class internally manages the aggregation of different datasets and allows transparent indexing across them.
#     However, it requires all constituent datasets to be fully loaded into memory, which might not be optimal for
#     very large datasets.

#     Upon initialization, this class computes the cumulative length of all datasets and maintains an internal mapping
#     of indices to the respective datasets. This approach allows the :class:`~torchtune.datasets.ConcatDataset`
#     to delegate data retrieval to the appropriate sub-dataset transparently when a particular index is accessed.

#     Note:
#         Using this class with very large datasets can lead to high memory consumption, as it requires all datasets to
#         be loaded into memory. For large-scale scenarios, consider other strategies that might stream data on demand.

#     Args:
#         datasets (List[Dataset]): A list of datasets to concatenate. Each dataset must be an instance of a class
#             derived from :class:`~torch.utils.data.Dataset`.

#     Raises
#     ------
#         ValueError: if instanse of `PackedDataset` is in `datasets`

#     Examples
#     --------
#         >>> dataset1 = MyCustomDataset(params1)
#         >>> dataset2 = MyCustomDataset(params2)
#         >>> concat_dataset = ConcatDataset([dataset1, dataset2])
#         >>> print(len(concat_dataset))  # Total length of both datasets
#         >>> data_point = concat_dataset[1500]  # Accesses an element from the appropriate dataset

#     This can also be accomplished by passing in a list of datasets to the YAML config::

#         dataset:
#           - _component_: torchtune.datasets.instruct_dataset
#             source: vicgalle/alpaca-gpt4
#             template: torchtune.data.AlpacaInstructTemplate
#             split: train
#             train_on_input: True
#           - _component_: torchtune.datasets.instruct_dataset
#             source: samsum
#             template: torchtune.data.SummarizeTemplate
#             column_map: {"output": "summary"}
#             output: summary
#             split: train
#             train_on_input: False

#     This class primarily focuses on providing a unified interface to access elements from multiple datasets,
#     enhancing the flexibility in handling diverse data sources for training machine learning models.
#     """

#     def __init__(self, datasets: List[Dataset]):
#         self._datasets: List[Dataset] = datasets

#         for dataset in self._datasets:
#             if isinstance(dataset, PackedDataset):
#                 raise ValueError("ConcatDataset can't process instances of PackedDataset.")

#         self._len: int = sum(len(dataset) for dataset in datasets)
#         self._indexes: List[Tuple[int, int, int]] = []

#         # Calculate distribution of indexes in all datasets
#         cumulative_index = 0
#         for idx, dataset in enumerate(datasets):
#             next_cumulative_index = cumulative_index + len(dataset)
#             self._indexes.append((cumulative_index, next_cumulative_index, idx))
#             cumulative_index = next_cumulative_index

#     def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
#         for start, stop, dataset_index in self._indexes:
#             if start <= index < stop:
#                 dataset = self._datasets[dataset_index]
#                 return dataset[index - start]

#     def __len__(self) -> int:
#         return self._len


# def get_lr(optimizer: Union[torch.optim.Optimizer, OptimizerInBackwardWrapper]) -> float:
#     """
#     Full_finetune_distributed and full_finetune_single_device assume all optimizers have
#     the same LR, here to validate whether all the LR are the same and return if True.

#     Args:
#         optimizer (Union[torch.optim.Optimizer, OptimizerInBackwardWrapper]): A general
#             optimizer input that could whether be a general optimizer or an optimizer
#             warpper based on optimizer_in_backward.

#     Returns
#     -------
#         lr (float): The learning rate of the input optimizers.

#     Raises
#     ------
#         RuntimeError: If the learning rates of the input optimizer are not the same.
#     """
#     if isinstance(optimizer, OptimizerInBackwardWrapper):
#         param_groups = []
#         for param in optimizer.state_dict().values():
#             param_groups.append(param["param_groups"][0])
#     else:
#         param_groups = optimizer.param_groups
#     if len(param_groups) < 1:
#         raise RuntimeError(f"Invalid optimizer param groups with len of: {len(param_groups)}")

#     # LR Schedulers are the same across all param groups for full_finetune right now
#     lr = param_groups[0]["lr"]
#     for group in param_groups:
#         if group["lr"] != lr:
#             raise RuntimeError("LR Schedulers are different across all param groups ")
#     return lr


# class FTRecipeInterface(Protocol):
#     """
#     This class provides a loose structure which every LLM fine-tuning recipe
#     should follow. Please note that the interface itself should not be a vehicle for
#     code reuse. torchtune strictly prohibits implementation inheritance in the codebase.

#     A few notes about the design and the need for this interface:
#     - This interface is meant to help recipe-writers organize their code in a way
#         which is easy to read, understand and extend. Minimizing code duplication is not
#         the goal. Recipe-writers are encouraged to copy-paste-modify.

#     - This interface is not meant to add constraints. If the interface comes in the
#         way of doing stuff, it needs to be updated or a new interface should be
#         written to support what might be a new "family" of recipes.
#     """

#     def load_checkpoint(self, **kwargs) -> None:
#         """
#         Responsible for loading ALL of the state for the recipe from the
#         checkpoint file, including state for the model, optimizer, dataloader and training
#         parameters such as the epoch and seed.
#         """
#         ...

#     def setup(self, **kwargs) -> None:
#         """
#         Responsible for setting up all of the components necessary for training. This includes
#         model, optimizer, loss function and dataloader.
#         """
#         ...

#     def train(self, **kwargs) -> None:
#         """
#         All of the training logic, including the core loop, loss computation, gradient
#         accumulation, and backward.
#         """
#         ...

#     def save_checkpoint(self, **kwargs) -> None:
#         """
#         Responsible for saving ALL of the state for the recipe,
#         including state for the model, optimizer, dataloader and training
#         parameters such as the epoch and seed.
#         """
#         ...

#     def cleanup(self, **kwargs) -> None:
#         """
#         Any cleaning up needed for the recipe.
#         """
#         ...


# class FullFinetuneRecipeSingleDevice(FTRecipeInterface):
#     """
#     Full finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe is optimized
#     for single GPU training. Training on CPU is not supported.

#     Features:
#         - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
#             flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
#             activations in memory and instead recompute them during the backward pass. This is especially
#             helpful for larger batch sizes when you're memory constrained. But these savings in memory
#             come at the cost of training performance. In most cases training can slow-down quite a bit as
#             a result of this activation recomputation.

#         - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
#             flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
#             most cases this should halve the memory footprint of full precision (fp32) training, without
#             loss in model quality (will depend on the model, training data and other settings). For
#             GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
#             precision are currently not supported.

#         - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
#             controlled using the ``gradient_accumulation_steps`` flag.

#                 Total Batch Size = batch_size * gradient accumulation steps.

#             For example: with batch_size=1 and gradient_accumulation_steps=32 we get a total batch size of 32.

#             Gradient accumulation is especially useful when you are memory constrained. In this case,
#             accumulating gradients might give you better training speed than enabling activation
#             checkpointing.

#         - Optimizer in Backward. Fusing the optimizer step into the backward pass helps reduce the memory
#             footprint associated with gradients. This can be especially helpful when you are memory
#             constrained. Note that users can only use ONE of gradient accumulation or optimizer in backward.
#             These features currently do not work together. For more details on optimizer in backward, please
#             see this tutorial: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html

#         - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
#             library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
#             8-bit AdamW and Paged AdamW. These optimizers are especially helpful when you are memory constrained
#             since they help reduce the memory footprint associated with the optimizer states.

#         - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
#             training. Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
#             only saved at the end of a given epoch and used in case of resuming training.

#             Resuming training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
#             currently not supported.

#             For more details on the checkpointer, please take a look at
#             our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

#         - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

#         - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
#             ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
#             ``clip_grad_norm='inf'``.

#     For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
#     has example commands for how to kick-off training.

#     Args:
#         cfg (DictConfig): OmegaConf object parsed from yaml file

#     Raises
#     ------
#         ValueError: If ``dtype`` is set to fp16.
#         RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
#         RuntimeError: If ``gradient_accumulation_steps > 1`` and ``optimizer_in_bwd`` is `True`.
#         RuntimeError: If ``left_pad_sequence`` is set as the data collator.
#     """

#     def __init__(self, cfg: DictConfig) -> None:
#         self._device = utils.get_device(device=cfg.device)
#         self._dtype = training.get_dtype(cfg.dtype, device=self._device)
#         # Disable for fp16, as we haven't validated "full" fp16 with this recipe, nor
#         # enabled necessary features such as gradient scaling.
#         if self._dtype == torch.float16:
#             raise ValueError("full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead.")

#         # logging attributes
#         self._output_dir = cfg.output_dir
#         self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
#         self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

#         # Training cfg
#         self._resume_from_checkpoint = cfg.resume_from_checkpoint
#         self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
#         self._optimizer_in_bwd = cfg.optimizer_in_bwd

#         # TODO: find a better place / way to perform validation of args that don't yet
#         # compose with each other.
#         if self._gradient_accumulation_steps > 1 and self._optimizer_in_bwd:
#             raise RuntimeError(
#                 "Gradient accumulation is not supported with optimizer in bwd."
#                 "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
#             )

#         # These are public properties which are updated by the checkpoint loader
#         # when ``resume_from_checkpoint`` is `True` or validated in tests
#         self.seed = training.set_seed(seed=cfg.seed)
#         self.epochs_run = 0
#         self.total_epochs = cfg.epochs
#         self.max_steps_per_epoch = cfg.max_steps_per_epoch
#         self.global_step = 0
#         self._clip_grad_norm = cfg.get("clip_grad_norm", None)

#     def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
#         """
#         Extract the checkpoint state from file and validate. If resume_from_checkpoint
#         is True, this also includes the recipe state.
#         """
#         self._checkpointer = config.instantiate(
#             cfg_checkpointer,
#             resume_from_checkpoint=self._resume_from_checkpoint,
#         )
#         checkpoint_dict = self._checkpointer.load_checkpoint()

#         if self._resume_from_checkpoint:
#             self._update_recipe_state(checkpoint_dict)
#         return checkpoint_dict

#     def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
#         """
#         Updates the recipe state from checkpoint.
#         """
#         try:
#             self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

#             # on mismatch, warn the user and prevent the override
#             if self.seed != ckpt_dict[training.SEED_KEY]:
#                 warn(
#                     message=(
#                         "Config value for seed does not match the checkpoint value, "
#                         f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
#                     )
#                 )
#                 self.seed = ckpt_dict[training.SEED_KEY]
#             if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
#                 warn(
#                     message=(
#                         "Config value for max_steps_per_epoch does not match the checkpoint value, "
#                         f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
#                     )
#                 )
#                 self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

#             # on mismatch, warn the user but allow the override
#             if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
#                 warn(
#                     message=(
#                         "Config value for total_epochs does not match the checkpoint value, "
#                         f"using the config value: {self.total_epochs}"
#                     )
#                 )

#         except KeyError as e:
#             raise KeyError(
#                 "Checkpoint does not contain the required keys needed for updating recipe state. "
#                 "Are you sure you passed in the right recipe checkpoint?"
#             ) from e

#     def setup(self, cfg: DictConfig) -> None:
#         """
#         Sets up the recipe state correctly. This includes setting recipe attributes based
#         on the ``resume_from_checkpoint`` flag.
#         """
#         self._metric_logger = config.instantiate(cfg.metric_logger)

#         # log config with parameter override
#         self._metric_logger.log_config(cfg)

#         ckpt_dict = self.load_checkpoint(cfg.checkpointer)

#         # ``_setup_model`` handles initialization and loading the state dict. This method
#         # should be called before ``_setup_optimizer`` since transforming the optimizer
#         # state dict requires the model
#         self._compile = cfg.compile
#         self._model = self._setup_model(
#             cfg_model=cfg.model,
#             enable_activation_checkpointing=cfg.enable_activation_checkpointing,
#             compile_model=self._compile,
#             model_state_dict=ckpt_dict[training.MODEL_KEY],
#         )
#         self._tokenizer = config.instantiate(cfg.tokenizer)
#         log.info("Tokenizer is initialized from file.")

#         # _setup_optimizer should take in ckpt_dict only if training is resumed from
#         # checkpoint. Transforming the opt state dict is handled by this method
#         self._optimizer = self._setup_optimizer(
#             cfg_optimizer=cfg.optimizer,
#             optimizer_in_bwd=cfg.optimizer_in_bwd,
#             opt_state_dict=(ckpt_dict[training.OPT_KEY] if self._resume_from_checkpoint else None),
#         )

#         # initialize loss
#         self._loss_fn = config.instantiate(cfg.loss)

#         if self._compile:
#             training.compile_loss(self._loss_fn)

#         if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
#             # set num_output_chunks for model
#             self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

#         log.info("Loss is initialized.")

#         # sampler and dataloader depend on the tokenizer and loss_fn and should be
#         # setup after both of these are initialized
#         collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
#         self._sampler, self._dataloader = self._setup_data(
#             cfg_dataset=cfg.dataset,
#             shuffle=cfg.shuffle,
#             batch_size=cfg.batch_size,
#             collate_fn=collate_name,
#         )

#         # Finally update the recipe state which can only be correctly set after all of the
#         # other components have been initialized and updated.
#         #
#         # Number of training steps in each epoch depends on the number of batches produced
#         # by the dataloader, the max_steps_per_epoch param set by the user and the
#         # gradient_accumulation_steps param. This value is used for logging and tracking
#         # training state. The computation should happen after the dataloader has been setup
#         self._steps_per_epoch = len(self._dataloader) // self._gradient_accumulation_steps
#         if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < self._steps_per_epoch:
#             self._steps_per_epoch = self.max_steps_per_epoch
#         self.global_step = self.epochs_run * self._steps_per_epoch

#         # Setup lr scheduler
#         self._lr_scheduler = self._setup_lr_scheduler(
#             cfg_lr_scheduler=cfg.get("lr_scheduler", None),
#             num_training_steps=self.total_epochs * self._steps_per_epoch,
#             last_epoch=self.global_step - 1,
#         )

#         # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
#         # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
#         self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

#         # Used to ignore labels for loss computation
#         self.ignore_labels_cache = torch.full((cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device)

#     def _setup_profiler(
#         self, cfg_profiler: Optional[DictConfig] = None
#     ) -> Union[torch.profiler.profile, DummyProfiler]:
#         """
#         Parses the `profiler` section of top-level `cfg` and sets up profiler

#         Args:
#             cfg_profiler (Optional[DictConfig]): ``profiler`` section of the top-level ``cfg`` (the main config passed to
#                 `recipe.main`). Default None.

#         Returns
#         -------
#             profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
#             for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
#             that the instrumented training loop does not need to be changed profiling is disabled.

#         The profiler config can be provided in configs under the `profiler` key with the following layout:

#         .. code-block:: yaml
#             profiler:
#                 enabled: bool

#                 #Output directory of trace artifacts
#                 output_dir: str

#             #`torch.profiler.ProfilerActivity` types to trace
#             cpu: bool
#             cuda: bool

#                 #Trace options
#                 profile_memory: bool
#                 with_stack: bool
#                 record_shapes: bool
#                 with_flops: bool

#             # `torch.profiler.schedule` options:
#             # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
#             wait_steps: int
#             warmup_steps: int
#             active_steps: int
#             num_cycles: int
#         """
#         # Missing profiler section in config, assume disabled
#         if cfg_profiler is None:
#             cfg_profiler = DictConfig({"enabled": False})

#         # Check that component is included and set correctly
#         if cfg_profiler.get("_component_", None) is None:
#             cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
#         else:
#             assert (
#                 cfg_profiler.get("_component_") == "torchtune.training.setup_torch_profiler"
#             ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

#         profiler, profiler_cfg = config.instantiate(cfg_profiler)

#         log.info(f" Profiler config after instantiation: {profiler_cfg}")

#         self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
#         if profiler_cfg["enabled"]:
#             self.profiler_wait_steps = profiler_cfg["wait_steps"]
#             self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
#             self.profiler_active_steps = profiler_cfg["active_steps"]

#         return profiler

#     def _setup_model(
#         self,
#         cfg_model: DictConfig,
#         enable_activation_checkpointing: bool,
#         compile_model: bool,
#         model_state_dict: Dict[str, Any],
#     ) -> nn.Module:
#         """
#         Set up the model including enabling activation checkpointing.
#         """
#         with training.set_default_dtype(self._dtype), self._device:
#             model = config.instantiate(cfg_model)

#         if compile_model:
#             training.compile_model(model)

#         if enable_activation_checkpointing:
#             training.set_activation_checkpointing(model, auto_wrap_policy={modules.TransformerSelfAttentionLayer})

#         model.load_state_dict(model_state_dict)

#         # Validate model was loaded in with the expected dtype.
#         training.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
#         log.info(f"Model is initialized with precision {self._dtype}.")

#         if self._device.type == "cuda":
#             memory_stats = training.get_memory_stats(device=self._device)
#             training.log_memory_stats(memory_stats)

#         return model

#     def _setup_optimizer(
#         self,
#         cfg_optimizer: DictConfig,
#         optimizer_in_bwd: bool = False,
#         opt_state_dict: Optional[Dict[str, Any]] = None,
#     ) -> Optional[Optimizer]:
#         """
#         Set up the optimizer. This method also handles loading the optimizer state_dict, if specified.
#         """
#         if optimizer_in_bwd:
#             # Maintain a dict of optims for every parameter.
#             optim_dict = {p: config.instantiate(cfg_optimizer, [p]) for p in self._model.parameters()}
#             # Register optimizer step hooks on the model to run optimizer in backward.
#             training.register_optim_in_bwd_hooks(model=self._model, optim_dict=optim_dict)
#             # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
#             self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(model=self._model, optim_dict=optim_dict)
#             # Load optimizer states. If optimizer states are being restored in an optimizer in backward
#             # run, these need to have been saved with the same setting. Cannot restore from runs that did not
#             # use optimizer in backward.
#             if opt_state_dict is not None:
#                 try:
#                     self._optim_ckpt_wrapper.load_state_dict(opt_state_dict)
#                 except BaseException as e:
#                     raise RuntimeError(
#                         "Failed loading in-backward optimizer checkpoints."
#                         "Please make sure run being restored from was using in-backward optimizer."
#                     ) from e
#             log.info("In-backward optimizers are set up.")
#             return None
#         else:
#             optimizer = config.instantiate(cfg_optimizer, self._model.parameters())

#             if opt_state_dict:
#                 optimizer.load_state_dict(opt_state_dict)
#             log.info("Optimizer is initialized.")
#             return optimizer

#     def _setup_lr_scheduler(
#         self,
#         cfg_lr_scheduler: Optional[DictConfig],
#         num_training_steps: int,
#         last_epoch: int,
#     ) -> Optional[Optimizer]:
#         """
#         Set up the learning rate scheduler based on the provided configuration.
#         It handles both standard optimization and optimizer-in-backward cases, and supports
#         schedulers from both torchtune.modules and torch.optim.

#         Args:
#             cfg_lr_scheduler (Optional[DictConfig]): The learning rate scheduler configuration.
#             num_training_steps (int): The total number of training steps.
#             last_epoch (int): The index of the last epoch.

#         Returns
#         -------
#             lr_scheduler (Optional[Optimizer]): The learning rate scheduler.
#         """
#         if cfg_lr_scheduler is None:
#             log.info("No learning rate scheduler configured. Using constant learning rate.")
#             return None

#         if self._optimizer_in_bwd:
#             # Use the first optimizer from the wrapper to represent the learning rate
#             optimizer = next(iter(self._optim_ckpt_wrapper.optim_map.values()))
#         else:
#             # Standard case: use the single optimizer
#             optimizer = self._optimizer

#         # Instantiate the learning rate scheduler
#         lr_scheduler = config.instantiate(
#             cfg_lr_scheduler,
#             optimizer,
#             num_training_steps=num_training_steps,
#             last_epoch=last_epoch,
#         )

#         if self._optimizer_in_bwd:
#             # Modify the scheduler for optimizer_in_bwd case
#             self._optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)

#         log.info("Learning rate scheduler is initialized.")
#         return lr_scheduler

#     def _setup_data(
#         self,
#         cfg_dataset: DictConfig,
#         shuffle: bool,
#         batch_size: int,
#         collate_fn: str,
#     ) -> Tuple[DistributedSampler, DataLoader]:
#         """
#         All data related setup happens here. Currently this recipe only supports the
#         DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
#         iterable datasets and streaming datasets are not supported.
#         """
#         if isinstance(cfg_dataset, ListConfig):
#             datasets = [config.instantiate(single_cfg_dataset, self._tokenizer) for single_cfg_dataset in cfg_dataset]
#             ds = ConcatDataset(datasets=datasets)
#             packed = False
#         else:
#             ds = config.instantiate(cfg_dataset, self._tokenizer)
#             packed = cfg_dataset.get("packed", False)

#         # Instantiate collate_fn
#         if "left_pad_sequence" in collate_fn:
#             raise RuntimeError("left_pad_sequence collator is only for inference.")
#         collate_fn = _get_component_from_path(collate_fn)

#         sampler = DistributedSampler(
#             ds,
#             num_replicas=1,
#             rank=0,
#             shuffle=shuffle,
#             seed=0,
#         )
#         dataloader = DataLoader(
#             dataset=ds,
#             batch_size=batch_size,
#             sampler=sampler,
#             # dropping last avoids shape issues with compile + flex attention
#             drop_last=True,
#             collate_fn=(
#                 partial(
#                     collate_fn,
#                     padding_idx=self._tokenizer.pad_id,
#                     ignore_idx=self._loss_fn.ignore_index,
#                 )
#                 if not packed
#                 else padded_collate_packed
#             ),
#         )

#         log.info("Dataset and Sampler are initialized.")

#         return sampler, dataloader

#     def save_checkpoint(self, epoch: int) -> None:
#         """
#         Save state dict to file. The recipe save_checkpoint method is responsible for
#         correctly creating the checkpoint dict and passing to the checkpointer.
#         """
#         ckpt_dict = {training.MODEL_KEY: self._model.state_dict()}
#         # if training is in-progress, checkpoint the optimizer state as well
#         if epoch + 1 < self.total_epochs:
#             ckpt_dict.update(
#                 {
#                     training.SEED_KEY: self.seed,
#                     training.EPOCHS_KEY: self.epochs_run,
#                     training.TOTAL_EPOCHS_KEY: self.total_epochs,
#                     training.MAX_STEPS_KEY: self.max_steps_per_epoch,
#                 }
#             )
#             if not self._optimizer_in_bwd:
#                 ckpt_dict[training.OPT_KEY] = self._optimizer.state_dict()
#             else:
#                 ckpt_dict[training.OPT_KEY] = self._optim_ckpt_wrapper.state_dict()
#         self._checkpointer.save_checkpoint(
#             ckpt_dict,
#             epoch=epoch,
#             intermediate_checkpoint=(epoch + 1 < self.total_epochs),
#         )

#     def _loss_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
#         # Shape [b, s], needed for the loss not the model
#         labels = batch.pop("labels")

#         logits = self._model(**batch)

#         # Shift labels to compute loss
#         # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
#         # But this way we dont need to slice the logits. We just add an ignore index to labels.
#         labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]]))
#         if not isinstance(logits, list):
#             labels = labels.reshape(-1)
#             logits = logits.reshape(-1, logits.size(-1))

#         # Compute loss
#         loss = self._loss_fn(logits, labels)
#         # free logits otherwise it peaks backward memory
#         del logits

#         return loss

#     def train(self) -> None:
#         """
#         The core training loop. Supports training on subsets of the dataset using the
#         ``max_steps_per_epoch``.
#         """
#         if self._compile:
#             log.info(
#                 "NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration."
#             )
#         # zero out the gradients before starting training
#         if not self._optimizer_in_bwd:
#             self._optimizer.zero_grad()

#         # Initialize tokens count and running loss (for grad accumulation)
#         t0 = time.perf_counter()
#         running_loss = 0
#         num_tokens = 0

#         self._profiler.start()
#         # self.epochs_run should be non-zero when we're resuming from a checkpoint
#         for curr_epoch in range(self.epochs_run, self.total_epochs):
#             # Update the sampler to ensure data is correctly shuffled across epochs
#             # in case shuffle is True
#             self._sampler.set_epoch(curr_epoch)

#             pbar = tqdm(total=self._steps_per_epoch)
#             for idx, batch in enumerate(self._dataloader):
#                 if (
#                     self.max_steps_per_epoch is not None
#                     and (idx // self._gradient_accumulation_steps) == self.max_steps_per_epoch
#                 ):
#                     break

#                 # Start tracking CUDA memory for active steps for just the first epoch
#                 if (
#                     curr_epoch == 0
#                     and self.profiler_profile_memory
#                     and idx == self.profiler_wait_steps + self.profiler_warmup_steps
#                 ):
#                     torch.cuda.memory._record_memory_history()

#                 utils.batch_to_device(batch, self._device)
#                 num_tokens += batch["tokens"].numel()

#                 loss = self._loss_step(batch)
#                 loss = loss / self._gradient_accumulation_steps
#                 running_loss += loss
#                 loss.backward()

#                 # Step with optimizer
#                 if (idx + 1) % self._gradient_accumulation_steps == 0:
#                     if self._clip_grad_norm is not None:
#                         grad_norm = torch.nn.utils.clip_grad_norm_(
#                             self._model.parameters(),
#                             max_norm=float(self._clip_grad_norm),
#                         )
#                     if not self._optimizer_in_bwd:
#                         self._optimizer.step()
#                         self._optimizer.zero_grad(set_to_none=True)

#                     # Need to fix `lr_scheduler.step()` before `optimizer.step()` warning
#                     if self._lr_scheduler is not None:
#                         self._lr_scheduler.step()
#                     self.global_step += 1

#                     loss_to_log = running_loss.item()
#                     pbar.update(1)
#                     pbar.set_description(f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}")

#                     # Log per-step metrics
#                     if self.global_step % self._log_every_n_steps == 0:
#                         time_per_step = time.perf_counter() - t0
#                         log_dict = {
#                             "loss": loss_to_log,
#                             # NOTE: for optim in backward, this assumes all optimizers have the same LR. This is currently
#                             # true since we don't expose the ability to configure this yet.
#                             "lr": get_lr(
#                                 self._optimizer if not self._optimizer_in_bwd else self._optim_ckpt_wrapper,
#                             ),
#                             "tokens_per_second_per_gpu": num_tokens / time_per_step,
#                         }
#                         if self._device.type == "cuda" and self._log_peak_memory_stats:
#                             log_dict.update(training.get_memory_stats(device=self._device))
#                         if self._clip_grad_norm is not None:
#                             log_dict.update({"grad_norm": grad_norm})
#                         self._metric_logger.log_dict(
#                             log_dict,
#                             step=self.global_step,
#                         )

#                     # Reset running stats for the next step
#                     running_loss = 0
#                     num_tokens = 0
#                     t0 = time.perf_counter()

#                 # Stop tracking CUDA memory now that active steps are complete
#                 if (
#                     curr_epoch == 0
#                     and self.profiler_profile_memory
#                     and idx == self.profiler_wait_steps + self.profiler_warmup_steps + self.profiler_active_steps
#                 ):
#                     torch.cuda.memory._record_memory_history(enabled=None)

#                 # Step the profiler
#                 # Note we are stepping each batch, which might not include optimizer step in the trace
#                 # if the schedule cycle doesn't align with gradient accumulation.
#                 self._profiler.step()

#             self.epochs_run += 1
#             self.save_checkpoint(epoch=curr_epoch)

#         self._profiler.stop()

#     def cleanup(self) -> None:
#         self._metric_logger.close()


# @config.parse
# def recipe_main(cfg: DictConfig) -> None:
#     """
#     Entry point for the recipe.

#     Configurable parameters are read in the following order:
#         - Parameters specified in config (see available configs through ``tune ls``)
#         - Overwritten by arguments from the command-line
#     """
#     config.log_config(recipe_name="FullFinetuneRecipeSingleDevice", cfg=cfg)
#     recipe = FullFinetuneRecipeSingleDevice(cfg=cfg)
#     recipe.setup(cfg=cfg)
#     recipe.train()
#     recipe.cleanup()


# if __name__ == "__main__":
#     sys.exit(recipe_main())
