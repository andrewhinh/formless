import copy
import functools
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache, partial
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, List, Literal, Mapping, Optional, Protocol, Set, Tuple, Union

import PIL
import torch
import torch.nn.functional as F

# from torchvision.transforms.v2 import functional as F
import torchvision
from PIL import Image
from tiktoken import Encoding
from tiktoken.load import load_tiktoken_bpe
from torch import distributed as dist
from torch import nn
from torch.distributed._tensor import DTensor, distribute_tensor
from torchao.dtypes.nf4tensor import linear_nf4, to_nf4

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

# data format


Role = Literal[
    "system",  # Origin is system prompt
    "user",  # Origin is user
    "assistant",  # Origin is the model output
    "ipython",  # Origin is return from a tool call
]
_TemplateType = Union[str, Dict[Role, Tuple[str, str]]]


class Message:
    """
    This class represents individual messages in a fine-tuning dataset. It supports
    text-only content, text with interleaved images, and tool calls. The :class:`~torchtune.modules.tokenizers.ModelTokenizer`
    will tokenize the content of the message using ``tokenize_messages`` and attach
    the appropriate special tokens based on the flags set in this class.

    Args:
        role (Role): role of the message writer. Can be "system" for system prompts,
            "user" for human prompts, "assistant" for model responses, or "ipython"
            for tool call returns.
        content (Union[str, List[Dict[str, Any]]]): content of the message. If it is text only content,
            you can pass in a string. If it is multimodal content, pass in a list of dictionaries formatted
            as follows::

                [
                    {"type": "image", "content": <PIL.Image.Image>},
                    {"type": "text", "content": "What is in this image?"},
                ]

        masked (bool): whether the message is masked in the sample. If True, do not use
            in loss calculation. Default: False
        ipython (bool): whether the message is a tool call. Default: False
        eot (bool): whether the message corresponds to the end of a turn, where control is handed over
            to the assistant from the user or the user from the assistant. Default: True. Should be true
            in most cases except for:

            - For multiple consecutive assistant messages (i.e., tool calls
              by assistant), only the last assistant message will have ``eot=True``
            - All ipython messages (tool call returns) should set ``eot=False``.

    Note:
        Message class expects any image content to be in
        `PIL Image format <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image>`_.
    """

    def __init__(
        self,
        role: Role,
        content: Union[str, List[Dict[str, Any]]],
        masked: bool = False,
        ipython: bool = False,
        eot: bool = True,
    ):
        self.role = role
        self.content = self._convert_to_list_of_dict(content)
        self.masked = masked
        self.ipython = ipython
        self.eot = eot

        self._validate_message()

    def _convert_to_list_of_dict(self, content) -> List[Dict[str, Any]]:
        """User is currently allowed to pass in a string for text-only content.
        This ensures that the content is formatted as a list of dictionaries.
        """
        if isinstance(content, str):
            return [{"type": "text", "content": content}]

        assert isinstance(content, list), f"content must be of type List[Dict[str, Any]], got {content}"

        return content

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        """
        Construct a Message from a dictionary.

        Args:
            d (dict): dictionary containing the fields of the Message.

        Returns
        -------
            Message: constructed Message.
        """
        return cls(
            role=d["role"],
            content=d["content"],
            masked=d.get("masked", False),
            ipython=d.get("ipython", False),
            eot=d.get("eot", True),
        )

    def get_media(self) -> List["PIL.Image.Image"]:
        """
        Returns media content of the message.
        """
        return [content["content"] for content in self.content if content["type"] == "image"]

    @property
    def contains_media(self) -> bool:
        """
        Returns whether the message contains media.
        """
        return any(content["type"] == "image" for content in self.content)

    @property
    def text_content(self) -> str:
        """
        Returns text-only content of the message.
        """
        return "".join(content["content"] for content in self.content if content["type"] == "text")

    def _validate_message(self) -> None:
        if self.ipython and self.contains_media:
            raise ValueError(
                f"Media tokens in tool calls are not supported. Both are set in message: {self.text_content}"
            )
        if self.ipython and self.role != "assistant":
            raise ValueError(
                f"Only assistant messages can be tool calls. Found role {self.role} in message: {self.text_content}"
            )

    def __repr__(self) -> str:
        content_only = [content["content"] for content in self.content]
        return f"Message(role='{self.role}', content={content_only!r})"


def truncate(
    tokens: List[Any],
    max_seq_len: int,
    eos_id: Optional[Any] = None,
) -> List[Any]:
    """
    Truncate a list of tokens to a maximum length. If eos_id is provided, the last
    token will be replaced with eos_id.

    Args:
        tokens (List[Any]): list of tokens to truncate
        max_seq_len (int): maximum length of the list
        eos_id (Optional[Any]): token to replace the last token with. If None, the
            last token will not be replaced. Default is None.

    Returns
    -------
        List[Any]: truncated list of tokens
    """
    tokens_truncated = tokens[:max_seq_len]
    if eos_id is not None and tokens_truncated[-1] != eos_id:
        tokens_truncated[-1] = eos_id
    return tokens_truncated


class PromptTemplateInterface(Protocol):
    """
    Interface for prompt templates. Each prompt template can include structured
    text for system, user, and assistant roles that are prepended or appended to
    the message content.
    """

    # Template should map role to a tuple containing the tag to prepend to the text
    # and tag to append to the text. Leave as empty strings to not prepend or append
    template: Dict[Role, Tuple[str, str]]

    def __call__(
        self,
        messages: List[Message],
        inference: bool = False,
    ) -> List[Message]:
        """
        Format each role's message(s) according to the prompt template

        Args:
            messages (List[Message]): a single conversation, structured as a list
                of :class:`~torchtune.data.Message` objects
            inference (bool): Whether the template is being used for inference or not.

        Returns
        -------
            The formatted list of messages
        """
        pass


class PromptTemplate(PromptTemplateInterface):
    r"""
    Quickly define a custom prompt template by passing in a dictionary mapping role to
    the prepend and append tags. For example, to achieve the following prompt
    template::

        System: {content}\\n
        User: {content}\\n
        Assistant: {content}\\n
        Tool: {content}\\n

    You need to pass in a tuple for each role, where ``PREPEND_TAG`` is the string
    added before the text content and ``APPEND_TAG`` is the string added after::

        template = {role: (PREPEND_TAG, APPEND_TAG)}

    Thus, the template would be defined as follows::

        template = {
            "system": ("System: ", "\\n"),
            "user": ("User: ", "\\n"),
            "assistant": ("Assistant: ", "\\n"),
            "ipython": ("Tool: ", "\\n"),
        }

    Once instantiated, you must call the prompt template on a list of messages. It
    will return the same list of messages updated with the template.

    Note:
        Any tags prepended/appended to the assistant message will be included
        in the loss calculation. All other prepend/append tags for other roles
        (system, user, ipython) are, in most cases, not included in loss. Consider using
        the append tags for user messages for tags that need to come before the
        assistant message but should not be included in loss. For more custom masking
        and prompt templating, you can create your own class based off the
        :class:`~torchtune.data.PromptTemplate` interface.

    Args:
        template (Dict[Role, Tuple[str, str]]): a dictionary mapping role to the
            prepend and append tags
    """

    def __init__(
        self,
        template: Dict[Role, Tuple[str, str]],
    ):
        self.template = template

    def __call__(self, messages: List[Message], inference: bool = False) -> List[Message]:
        """
        Format each role's message(s) according to the prompt template by prepending
        and appending the defined tags.

        Args:
            messages (List[Message]): list of messages to apply the template to
            inference (bool): Whether the template is being used for inference or not.

        Returns
        -------
            List[Message]: The formatted list of messages
        """
        formatted_dialogue = []
        for message in messages:
            content = message.content
            if message.role in self.template:
                prepend_tag = self.template[message.role][0]
                append_tag = self.template[message.role][1]
                content = message.content

                if isinstance(prepend_tag, str) and len(prepend_tag) > 0:
                    content = [{"type": "text", "content": prepend_tag}] + content

                if isinstance(append_tag, str) and len(append_tag) > 0:
                    content = content + [{"type": "text", "content": append_tag}]
            formatted_dialogue.append(
                Message(
                    role=message.role,
                    content=content,
                    masked=message.masked,
                    ipython=message.ipython,
                    eot=message.eot,
                ),
            )
        return formatted_dialogue


class ChatMLTemplate(PromptTemplateInterface):
    """
    OpenAI's `Chat Markup Language
    <https://github.com/MicrosoftDocs/azure-docs/blob/772c14eeabfa0c0c561d5c2d34ef19341f528b7b/articles/ai-services/openai/how-to/chat-markup-language.md>`_
    used by their chat models.

    It is the default chat template used by Hugging Face models.

    .. code-block:: text

        <|im_start|>system
        Provide some context and/or instructions to the model.<|im_end|>
        <|im_start|>user
        The user’s message goes here<|im_end|>
        <|im_start|>assistant
        The assistant’s response goes here<|im_end|>

    """

    template = {
        "system": ("<|im_start|>system\n", "<|im_end|>\n"),
        "user": ("<|im_start|>user\n", "<|im_end|>\n"),
        "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        "ipython": ("", ""),
    }

    def __call__(
        self,
        messages: List[Message],
        inference: bool = False,
    ) -> List[Message]:
        """
        Format user, assistant, and system messages with appropriate tags.

        Args:
            messages (List[Message]): a single conversation, structured as a list
                of `Message` objects
            inference (bool): Whether the template is being used for inference or not.

        Returns
        -------
            The formatted list of messages
        """
        formatted_dialogue = []
        for index, message in enumerate(messages):
            prepend_tag = self.template[message.role][0]
            append_tag = self.template[message.role][1]
            # If empty assistant message at the end, we are expecting the model
            # to generate the response continuing from the assistant prepend tag,
            # so do not add the append tag.
            if message.role == "assistant" and index == len(messages) - 1 and len(message.text_content) == 0:
                content = message.content
                if isinstance(prepend_tag, str) and len(prepend_tag) > 0:
                    content = [{"type": "text", "content": prepend_tag}] + message.content
            else:
                content = message.content

                if isinstance(prepend_tag, str) and len(prepend_tag) > 0:
                    content = [{"type": "text", "content": prepend_tag}] + content

                if isinstance(append_tag, str) and len(append_tag) > 0:
                    content = content + [{"type": "text", "content": append_tag}]

            formatted_dialogue.append(
                Message(
                    role=message.role,
                    content=content,
                    masked=message.masked,
                    ipython=message.ipython,
                    eot=message.eot,
                ),
            )
        return formatted_dialogue


GrammarErrorCorrectionTemplate = partial(
    PromptTemplate,
    template={
        "user": ("Correct this to standard English: ", "\n---\nCorrected: "),
    },
)
GrammarErrorCorrectionTemplate.__doc__ = """
A prompt template for grammar error correction tasks::

    Correct this to standard English: {user_message}
    ---
    Corrected: {assistant_message}

Please see :class:`~torchtune.data.PromptTemplate` for full API arguments.
"""
SummarizeTemplate = partial(
    PromptTemplate,
    template={
        "user": ("Summarize this dialogue:\n", "\n---\nSummary:\n"),
    },
)
SummarizeTemplate.__doc__ = """
A prompt template for summarization tasks::

    Summarize this dialogue:
    {user_message}
    ---
    Summary:
    {assistant_message}

Please see :class:`~torchtune.data.PromptTemplate` for full API arguments.
"""
QuestionAnswerTemplate = partial(
    PromptTemplate,
    template={
        "user": ("Question: ", "\n\nAnswer: "),
    },
)
QuestionAnswerTemplate.__doc__ = """
A prompt template for question answering tasks::

    Question: {user_message}

    Answer: {assistant_message}

Please see :class:`~torchtune.data.PromptTemplate` for full API arguments.
"""


class InstantiationError(Exception):
    """
    Raised when a `_component_` field in a config is unable to be instantiated.
    """

    pass


def _get_component_from_path(path: str) -> Any:
    """
    Return an object by name or dotted path, importing as necessary.
    The base functionality relies on ``getattr()`` and handles all
    possible exceptions accordingly.

    Based on Hydra's `_locate` from Facebook Research:
    https://github.com/facebookresearch/hydra/blob/main/hydra/_internal/utils.py#L614

    Args:
        path (str): Dotted path of the object

    Returns
    -------
        Any: The object

    Raises
    ------
        InstantiationError: If there is an exception loading the
            object from the provided path
        ValueError: If a relative or invalid dotpath is passed in
    """
    if path == "":
        raise ValueError("Empty path")

    parts = list(path.split("."))
    for part in parts:
        # If a relative path is passed in, the first part will be empty
        if not len(part):
            raise ValueError(f"Error loading '{path}': invalid dotstring." + "\nRelative imports are not supported.")
    # First module requires trying to import to validate
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except ImportError as exc_import:
        raise InstantiationError(
            f"Error loading '{path}':\n{repr(exc_import)}" + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    # Subsequent components can be checked via getattr() on first module
    # It can either be an attribute that we can return or a submodule that we
    # can import and continue searching
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        # If getattr fails, check to see if it's a module we can import and
        # continue down the path
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise InstantiationError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                # Any other error trying to import module can be raised as
                # InstantiationError
                except Exception as exc_import:
                    raise InstantiationError(f"Error loading '{path}':\n{repr(exc_import)}") from exc_import
            # If the component is not an attribute nor a module, it doesn't exist
            raise InstantiationError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj


def _get_prompt_template(
    prompt_template: _TemplateType,
) -> PromptTemplateInterface:
    """
    Retrieve prompt template from import dotpath or create a custom one with provided
    template dictionary.

    Args:
        prompt_template (_TemplateType): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.

    Returns
    -------
        PromptTemplateInterface: the specified prompt template

    Raises
    ------
        ValueError: If a string or dictionary is not passed in
    """
    if isinstance(prompt_template, str):
        return _get_component_from_path(prompt_template)()
    elif isinstance(prompt_template, dict):
        return PromptTemplate(prompt_template)
    else:
        raise ValueError(
            f"Prompt template must be a dotpath string or dictionary with custom template, got {type(prompt_template)}"
        )


# -----------------------------------------------------------------------------

# tokenizer


# Constants controlling encode logic
MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACE_CHARS = 25_000


class BaseTokenizer(Protocol):
    """
    Abstract token encoding model that implements ``encode`` and ``decode`` methods.
    See :class:`~torchtune.modules.tokenizers.SentencePieceBaseTokenizer` and
    :class:`~torchtune.modules.tokenizers.TikTokenBaseTokenizer` for example implementations of this protocol.
    """

    def encode(self, text: str, **kwargs: Dict[str, Any]) -> List[int]:
        """
        Given a string, return the encoded list of token ids.

        Args:
            text (str): The text to encode.
            **kwargs (Dict[str, Any]): kwargs.

        Returns
        -------
            List[int]: The encoded list of token ids.
        """
        pass

    def decode(self, token_ids: List[int], **kwargs: Dict[str, Any]) -> str:
        """
        Given a list of token ids, return the decoded text, optionally including special tokens.

        Args:
            token_ids (List[int]): The list of token ids to decode.
            **kwargs (Dict[str, Any]): kwargs.

        Returns
        -------
            str: The decoded text.
        """
        pass


class ModelTokenizer(Protocol):
    """
    Abstract tokenizer that implements model-specific special token logic in
    the ``tokenize_messages`` method. See :class:`~torchtune.models.llama3.Llama3Tokenizer`
    for an example implementation of this protocol.
    """

    special_tokens: Dict[str, int]
    max_seq_len: Optional[int]

    def tokenize_messages(self, messages: List[Message], **kwargs: Dict[str, Any]) -> Tuple[List[int], List[bool]]:
        """
        Given a list of messages, return a list of tokens and list of masks for
        the concatenated and formatted messages.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            **kwargs (Dict[str, Any]): kwargs.

        Returns
        -------
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        pass


class TikTokenBaseTokenizer(BaseTokenizer):
    """
    A lightweight wrapper around tiktoken Encoding. This class additionally handles
    breaking up the input text into substrings of a max length and splitting up long
    repetitions to improve encode speed.

    Args:
        path (str): Path to pretrained tokenizer checkpoint file.
        name (str): Name of the tokenizer (used by tiktoken for identification).
        pattern (str): Regex pattern used to split input text into chunks before passing
            to byte-pair encoding.
        bos_id (int): beginning-of-sequence token id. This can be present or absent in ``special_tokens``.
        eos_id (int): end-of-sequence token id. This can be present or absent in ``special_tokens``.
        special_tokens (Dict[str, int]): Mapping of special tokens to their ids.

    Examples
    --------
        >>> tokenizer = TikTokenBaseTokenizer("/path/to/tt_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        name: str,
        pattern: str,
        bos_id: int,
        eos_id: int,
        special_tokens: Dict[str, int],
    ):
        mergeable_ranks = load_tiktoken_bpe(path)
        self.tt_model = Encoding(
            name=name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        # Vocab size without special tokens
        self.base_vocab_size = len(mergeable_ranks)
        # Vocab size with special tokens
        self.vocab_size = self.tt_model.n_vocab
        self.bos_id = bos_id
        self.eos_id = eos_id

    def _split_long_repetitions(self, s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        """
        Split the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Encode a string into a list of token ids. Assumes that the string
        contains no special tokens.

        Args:
            text (str): The string to encode.
            add_bos (bool): Whether to add the tokenizer's bos_id to the encoded string.
                Default True.
            add_eos (bool): Whether to add the tokenizer's eos_id to the encoded string.
                Default True.

        Returns
        -------
            List[int]: The list of token ids.
        """
        substrs: List[str] = []
        tokens = []
        if not text:
            return []
        for i in range(0, len(text), MAX_ENCODE_CHARS):
            substr = text[i : i + MAX_ENCODE_CHARS]
            # See https://github.com/openai/tiktoken/issues/195
            sliced_substr = self._split_long_repetitions(substr, MAX_NO_WHITESPACE_CHARS)
            substrs.extend(sliced_substr)
        for substr in substrs:
            # allowed_special and disallowed_special are used by tiktoken to define
            # how special tokens are encoded. Our setting here is to encode any
            # special token as regular text and prevent tiktoken from raising errors.
            # This means we should only call encode on strings not containing special tokens.
            tokens.extend(
                self.tt_model.encode(
                    substr,
                    allowed_special=set(),
                    disallowed_special=(),
                )
            )
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            truncate_at_eos (bool): Whether to truncate the string at the end of
                sequence token. Default is True.

        Returns
        -------
            str: The decoded string.
        """
        if truncate_at_eos:
            try:
                k = token_ids.index(self.eos_id)
            except ValueError:
                k = None
            if k:
                token_ids = token_ids[:k]
        return self.tt_model.decode(token_ids)


def parse_hf_tokenizer_json(tokenizer_json_path: str) -> Dict[str, int]:
    """
    Parse the ``tokenizer.json`` file from a Hugging Face model to extract the
    special token str to id mapping.

    Args:
        tokenizer_json_path (str): Path to the ``tokenizer.json`` file.

    Returns
    -------
        Dict[str, int]: The special token str to id mapping.
    """
    with open(tokenizer_json_path, "r") as f:
        tokenizer_json = json.load(f)

    return {token["content"]: token["id"] for token in tokenizer_json["added_tokens"]}


CL100K_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa

SPECIAL_TOKENS = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|reserved_special_token_0|>": 128002,
    "<|reserved_special_token_1|>": 128003,
    "<|finetune_right_pad_id|>": 128004,
    "<|step_id|>": 128005,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eom_id|>": 128008,
    "<|eot_id|>": 128009,
    "<|python_tag|>": 128010,
    "<|image|>": 128256,
    "<|video|>": 128012,
}

NUM_RESERVED_SPECIAL_TOKENS = 256

RESERVED_TOKENS = {
    f"<|reserved_special_token_{2 + i}|>": 128013 + i for i in range(NUM_RESERVED_SPECIAL_TOKENS - len(SPECIAL_TOKENS))
}

LLAMA3_SPECIAL_TOKENS = {**SPECIAL_TOKENS, **RESERVED_TOKENS}


class Transform(Protocol):
    """
    Loose interface for all data and model transforms. Transforms operate at the
    sample level and perform operations on a sample dict, returning the updated dict.
    For an example implementation of this protocol, see
    :class:`~torchtune.modules.transforms.VisionCrossAttentionMask`.
    """

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        pass


class Llama3Tokenizer(ModelTokenizer, Transform):
    """
    tiktoken tokenizer configured with Llama3 Instruct's special tokens, as described in
    https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

    Args:
        path (str): Path to pretrained tiktoken tokenizer file.
        special_tokens (Optional[Dict[str, int]]): mapping containing special text tokens and
            their registered token IDs. If left as None, this will be set to the canonical
            Llama3 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens. Default is None.

    Examples
    --------
        >>> tokenizer = Llama3Tokenizer("/path/to/tt_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        special_tokens: Optional[Dict[str, int]] = None,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        self.special_tokens = special_tokens if special_tokens is not None else LLAMA3_SPECIAL_TOKENS

        self._validate_special_tokens()

        # Encode BOS and EOS, define pad ID
        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        self.pad_id = self.special_tokens["<|finetune_right_pad_id|>"]
        self.step_id = self.special_tokens["<|step_id|>"]

        # Encode extra special tokens
        self.start_header_id = self.special_tokens["<|start_header_id|>"]
        self.end_header_id = self.special_tokens["<|end_header_id|>"]
        self.eot_id = self.special_tokens["<|eot_id|>"]

        self.eom_id = self.special_tokens["<|eom_id|>"]
        self.python_tag = self.special_tokens["<|python_tag|>"]

        # Media tokens
        self.image_id = self.special_tokens["<|image|>"]

        # During generation, stop when either eos_id, eot_id, or eom_id is encountered
        self.stop_tokens = [self.eos_id, self.eot_id, self.eom_id]

        self.tt_model = TikTokenBaseTokenizer(
            path=path,
            name="llama3_tiktoken",
            pattern=CL100K_PATTERN,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            special_tokens=self.special_tokens,
        )
        self.max_seq_len = max_seq_len

        self.prompt_template = prompt_template

        # Regex for removing special tokens from the decoded string
        self._special_token_regex = re.compile(r"<\|.*?\|>")
        self._special_token_header_regex = re.compile(r"<\|start_header_id\|>.*?<\|end_header_id\|>\n\n")

    def _validate_special_tokens(
        self,
    ):
        """
        Validate that required special tokens are passed into the tokenizer.
        """
        for token in [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",
            "<|eot_id|>",
            "<|python_tag|>",
        ]:
            if token not in self.special_tokens:
                raise ValueError(f"{token} missing from special_tokens")

    def _remove_special_tokens(self, text: str) -> str:
        """
        Remove special tokens from the decoded string.
        """
        # First remove the headers, then the remaining special tokens
        return self._special_token_regex.sub("", self._special_token_header_regex.sub("", text))

    @property
    def base_vocab_size(self) -> int:
        return self.tt_model.base_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.tt_model.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        return self.tt_model.encode(text=text, add_bos=add_bos, add_eos=add_eos)

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            truncate_at_eos (bool): Whether to truncate the string at the end of
                sequence token. Default is True.
            skip_special_tokens (bool): Whether to show or skip special tokens in the decoded string.
                Default is True.

        Returns
        -------
            str: The decoded string.
        """
        # We will remove special tokens manually via regex on the decoded string.
        # This is because removing all special tokens does not remove the role and
        # whitespace added from the special tokens, i.e., the "user" and "\n\n" in
        # "<|start_header_id|>user<|end_header_id|>\n\n"
        decoded_string = self.tt_model.decode(
            token_ids=token_ids,
            truncate_at_eos=truncate_at_eos,
        )
        return self._remove_special_tokens(decoded_string) if skip_special_tokens else decoded_string

    def _tokenize_header(self, message: Message) -> List[int]:
        """
        Tokenize header start, message role, and header end as list of ids
        """
        return (
            [self.start_header_id]
            + self.encode(message.role.strip(), add_bos=False, add_eos=False)
            + [self.end_header_id]
            + self.encode("\n\n", add_bos=False, add_eos=False)
        )

    def _tokenize_end(self, message: Message) -> List[int]:
        """
        Add eot or eom id at the end of the message.
        """
        return [self.eot_id] if message.eot else [self.eom_id]

    def _tokenize_body(self, message: Message) -> List[int]:
        """
        Tokenize message content as list of ids
        """
        tokenized_body = []
        for item in message.content:
            if item["type"] == "text":
                tokenized_body += self.encode(item["content"].strip(), add_bos=False, add_eos=False)
            elif item["type"] == "image":
                tokenized_body += [self.image_id]
            else:
                raise RuntimeError(f"Unsupported message content type: {item['type']}")

        if message.ipython:
            tokenized_body = [self.python_tag] + tokenized_body

        return tokenized_body

    def tokenize_message(
        self,
        message: Message,
        *,
        add_start_tokens: bool = True,
        add_end_tokens: bool = True,
    ) -> List[int]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            add_start_tokens (bool): Whether to prepend a tokenized header to the message. Default is True.
            add_end_tokens (bool): Whether to append eot or eom id at the end of the message. Default is True.

        Returns
        -------
            List[int]: The list of token ids.
        """
        tokenized_header = self._tokenize_header(message) if add_start_tokens else []
        tokenized_body = self._tokenize_body(message)
        tokenized_end = self._tokenize_end(message) if add_end_tokens else []

        tokenized_message = tokenized_header + tokenized_body + tokenized_end
        return tokenized_message

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_end_tokens: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            add_end_tokens (bool): Whether to append end tokens ids (end-of-seq, end-of-turn, end-of-message) at the end of the
                last assistant message. This value should be set to False for generation. Default is True.

        Examples
        --------
            >>> # Tokenize a list of messages with default settings
            >>> messages = [
            ...     Message(role="user", content="Hello world!", masked=True),
            ...     Message(role="assistant", content="How are you?", masked=False),
            ... ]
            >>> tokenizer = Llama3Tokenizer("/path/to/tt_model")
            >>> tokenizer.tokenize_messages(messages)
            ([1, 31587, 29644, 102, 1, 31587, 29644, 102, 2], [True, True, True, True, True, False, False, False, True])

            >>> # Tokenize a list of messages with add_end_tokens set to False
            >>> tokenizer.tokenize_messages(messages, add_end_tokens=False)
            ([1, 31587, 29644, 102, 1, 31587, 29644], [True, True, True, True, True, False, False])

        Returns
        -------
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        templated_messages = self.prompt_template(messages) if self.prompt_template is not None else messages
        tokens = [self.bos_id]
        # bos and eos are always masked
        mask = [True]

        num_messages = len(templated_messages)
        for i, message in enumerate(templated_messages):
            # Add end tokens to the last assistant message if add_end_tokens is True
            # Otherwise, end tokens should always be added
            add_end_tokens_to_message = add_end_tokens if i == num_messages - 1 else True
            tokenized_message = self.tokenize_message(message, add_end_tokens=add_end_tokens_to_message)

            tokens = tokens + tokenized_message
            mask = mask + ([message.masked] * len(tokenized_message))
            if self.max_seq_len and len(tokens) >= self.max_seq_len:
                break

        if add_end_tokens:
            tokens = tokens + [self.eos_id]
            mask = mask + [True]

        if self.max_seq_len:
            tokens = truncate(tokens, self.max_seq_len, self.eos_id if add_end_tokens else None)
            mask = truncate(mask, self.max_seq_len, True if add_end_tokens else None)

        return tokens, mask

    def __call__(self, sample: Mapping[str, Any], inference: bool = False) -> Mapping[str, Any]:
        """
        Apply ``tokenize_messages`` to the "messages" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field containing
                a List[Message] to tokenize
            inference (bool): Whether the template is being used for inference or not.

        Returns
        -------
            Mapping[str, Any]: The sample with added "tokens" and "mask" fields
                and the "messages" field removed.
        """
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages, add_end_tokens=not inference)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample


def llama3_tokenizer(
    path: str,
    special_tokens_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[_TemplateType] = None,
) -> Llama3Tokenizer:
    """
    Tokenizer for Llama3.

    Args:
        path (str): path to the tokenizer
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Llama3 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.

    Returns
    -------
        Llama3Tokenizer: Instantiation of the Llama3 tokenizer
    """
    special_tokens = parse_hf_tokenizer_json(special_tokens_path) if special_tokens_path is not None else None
    template = _get_prompt_template(prompt_template) if prompt_template is not None else None
    return Llama3Tokenizer(path=path, special_tokens=special_tokens, max_seq_len=max_seq_len, prompt_template=template)


# -----------------------------------------------------------------------------

# layers


## base
class FeedForward(nn.Module):
    """This class implements the feed-forward network derived from Llama2.

    Args:
        gate_proj (nn.Module): Projection from input dim to hidden dim, fed through activation
            and multiplied by up_proj.
        down_proj (nn.Module): Final projection to output dim.
        up_proj (Optional[nn.Module]): Projection from input dim to hidden dim, multiplied by
            activation(gate_proj).
        activation (nn.Module): Activation function to use. Default is nn.SiLU().
    """

    def __init__(
        self,
        *,
        gate_proj: nn.Module,
        down_proj: nn.Module,
        up_proj: Optional[nn.Module] = None,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.w1 = gate_proj
        self.w2 = down_proj
        self.w3 = up_proj
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``, where ``in_dim`` is the
                input dimension of both ``gate_proj`` and ``up_proj``.

        Returns
        -------
            torch.Tensor: output tensor with shape ``(..., out_dim)``, where ``out_dim`` is the \
                output dimension of ``down_proj``.
        """
        h = self.activation(self.w1(x))
        if self.w3 is not None:
            h = h * self.w3(x)
        h = self.w2(h)
        return h


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/abs/1910.07467.

    Reference implementation (used for correctness verification)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns
        -------
            torch.Tensor: The normalized and scaled tensor having the same shape as ``x``.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return x_normed * self.scale


class TanhGate(nn.Module):
    """Implements a basic learnable gate to scale layer outputs"""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to gate

        Returns
        -------
            torch.Tensor: The output tensor after gating. Has the same shape as ``x``.
        """
        return x * self.scale.tanh()


class Fp32LayerNorm(nn.LayerNorm):
    """
    Wrapper around :class:`~torch.nn.LayerNorm` to support mixed-precision training.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: The normalized output tensor having the same shape as ``x``.
        """
        output = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class FrozenNF4Linear(nn.Linear):
    """
    A linear layer similar to ``torch.nn.Linear`` but uses a quantized
    NF4Tensor as its weight. This class also freezes its ``weight`` parameter
    and is meant to be used as the base Linear layer for modeling
    use cases such as QLoRA where base model parameters are frozen.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        device (Optional[torch.device]): device to use for the underlying weight. If ``None``, uses the default
            device given by `torch.get_default_device()`.
        bias (bool): whether to include bias in the linear layer. Default: False
        **kwargs: any additional arguments to pass to the underlying Linear layer.

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: Optional[torch.device] = None,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(in_dim, out_dim, device=device, bias=bias, **kwargs)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        nf4_weight = to_nf4(self.weight)
        # re-register self.weight as the nf4 weight, so that the nf4 weight
        # shows up as expected in .parameters, state_dict, etc.
        torch.utils.swap_tensors(self.weight, torch.nn.Parameter(nf4_weight, requires_grad=False))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Runs linear operation with input tensor as given by `input`. Computation happens in higher
        precision, though only the nf4 weight is saved for backward for gradient computation to ensure
        additional memory is not used.
        Args:
            input (torch.Tensor): input tensor

        Returns
        -------
            Tensor: output tensor
        """
        out = linear_nf4(input=input, weight=self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


## mlp
def scale_hidden_dim_for_mlp(dim: int, multiple_of: int = 256) -> int:
    """Scale hidden dimension for MLP to keep number of parameters and computation constant.

    Args:
        dim (int): Input dimension.
        multiple_of (int): Round scaled dimension to nearest multiple of `multiple_of` for clean computation.

    Returns
    -------
        Scaled hidden dimension.
    """
    # Scale hidden dimension by (2/3)4d for SwiGLU to keep number of
    # parameters and computation constant
    hidden_dim = 4 * int(2 * dim / 3)
    # Round hidden dimension to nearest multiple of `multiple_of`
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def llama3_mlp(dim: int, hidden_dim: int, quantize_base: bool = False) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = (
        nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    )
    down_proj = (
        nn.Linear(hidden_dim, dim, bias=False) if not quantize_base else FrozenNF4Linear(hidden_dim, dim, bias=False)
    )
    up_proj = (
        nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    )
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


## clip


class TokenPositionalEmbedding(nn.Module):
    """
    Token positional embedding for images, different for every token in an image.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(self, embed_dim: int, tile_size: int, patch_size: int) -> None:
        super().__init__()
        patch_grid_size = tile_size // patch_size
        n_tokens_per_tile = patch_grid_size**2 + 1  # +1 for cls token
        scale = embed_dim**-0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn((n_tokens_per_tile, embed_dim)))

    def forward(self, x: torch.Tensor, *args: Tuple[Any]) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape (..., n_tokens_per_tile, embed_dim)
            *args (Tuple[Any]): Optional args.

        Returns
        -------
            torch.Tensor: The input tensor with added positional embeddings.
        """
        return x + self.positional_embedding


class TiledTokenPositionalEmbedding(nn.Module):
    """

    Token positional embedding for tiled images, different for every tile, different for every token.

    There are two positional embeddings in this module:

    * local_token_positional_embedding: same for every tile, different for every token. Equivalent \
        to :class:`torchtune.models.clip._position_embeddings.TokenPositionalEmbedding`, but gated.
    * global_token_positional_embedding: different for every tile, different for every token.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(self, max_num_tiles: int, embed_dim: int, tile_size: int, patch_size: int) -> None:
        super().__init__()

        patch_grid_size = tile_size // patch_size
        self.n_tokens_per_tile = patch_grid_size**2 + 1  # +1 for cls token
        scale = embed_dim**-0.5

        # different for every token, same for every tile
        self.local_token_positional_embedding = nn.Parameter(scale * torch.randn((self.n_tokens_per_tile, embed_dim)))

        # different for every token, different for every tile
        self.global_token_positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                max_num_tiles,
                max_num_tiles,
                self.n_tokens_per_tile,
                embed_dim,
            )
        )

        self.gate = nn.Parameter(torch.zeros(1))

        self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

    @torch.no_grad()
    def _load_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Interpolates positional embeddings to accomodate different number of tiles
        and tokens per tile, in case the model was instantiated with different
        settings than the one you are loading the state dict from.

        For more info, please check self._resize_local_position_embedding and
        self._resize_global_position_embedding functions.

        Args:
            state_dict (Dict[str, Any]): The state dict to load.
            prefix (str): The prefix of the state dict.
            *args (Tuple[Any]): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Raises
        ------
            ValueError: if loaded local or global embedding n_tokens_per_tile is not derived
                from a squared grid.
            ValueError: if after interpolation, the shape of the loaded local embedding
                is not compatible with the current embedding.
            ValueError: if after interpolation, the shape of the loaded global embedding
                is not compatible with the current embedding.
        """
        # process local_token_positional_embedding
        inpt_local_pos_embed = state_dict.get(prefix + "local_token_positional_embedding")

        if inpt_local_pos_embed is not None:
            # We can only apply F.interpolate to vanilla tensors, not DTensors
            # If pos embeds are a DTensor, we gather the full tensor, apply
            # interpolate, and then reshard after
            if isinstance(inpt_local_pos_embed, DTensor):
                local_embed_is_sharded = True
                local_embed_device_mesh = inpt_local_pos_embed.device_mesh
                local_embed_placements = inpt_local_pos_embed.placements
                inpt_local_pos_embed = inpt_local_pos_embed.full_tensor()
            else:
                local_embed_is_sharded = False

            # sanity check
            inpt_n_tokens_per_tile, inpt_embed_dim = inpt_local_pos_embed.shape
            if math.sqrt(inpt_n_tokens_per_tile - 1) % 1 != 0:
                raise ValueError(
                    f"Loaded local positional embedding has shape {inpt_n_tokens_per_tile=}, "
                    f"which indicates a grid_size that is not squared. This is currently not supported."
                )

            # instantiated pos emb
            (
                tgt_n_tokens_per_tile,
                tgt_embed_dim,
            ) = self.local_token_positional_embedding.shape

            # resize ckpt to match instantiated shape
            inpt_local_pos_embed = self._resize_local_position_embedding(
                local_pos_embed=inpt_local_pos_embed,
                tgt_patch_grid_size=int(math.sqrt(tgt_n_tokens_per_tile - 1)),
            )

            if local_embed_is_sharded:
                inpt_local_pos_embed = distribute_tensor(
                    inpt_local_pos_embed,
                    device_mesh=local_embed_device_mesh,
                    placements=local_embed_placements,
                )

            # update state dict
            state_dict[prefix + "local_token_positional_embedding"] = inpt_local_pos_embed
            if inpt_local_pos_embed.shape != self.local_token_positional_embedding.shape:
                raise ValueError(
                    f"Loaded local positional embedding has shape {inpt_local_pos_embed.shape}, "
                    f"after interpolation. Expected shape {self.local_token_positional_embedding.shape}."
                )

        # process global_token_positional_embedding
        inpt_global_pos_embed = state_dict.get(prefix + "global_token_positional_embedding")

        if inpt_global_pos_embed is not None:
            # We can only apply F.interpolate to vanilla tensors, not DTensors
            # If pos embeds are a DTensor, we gather the full tensor, apply
            # interpolate, and then reshard after
            if isinstance(inpt_global_pos_embed, DTensor):
                global_embed_is_sharded = True
                global_embed_device_mesh = inpt_global_pos_embed.device_mesh
                global_embed_placements = inpt_global_pos_embed.placements
                inpt_global_pos_embed = inpt_global_pos_embed.full_tensor()
            else:
                global_embed_is_sharded = False

            _, _, inpt_n_tokens_per_tile, _ = inpt_global_pos_embed.shape

            # sanity check
            if math.sqrt(inpt_n_tokens_per_tile - 1) % 1 != 0:
                raise ValueError(
                    f"Loaded local positional embedding has shape {inpt_n_tokens_per_tile=}, "
                    f"which indicates a grid_size that is not squared. This is currently not supported."
                )

            # instantiated pos emb
            (
                tgt_max_num_tiles_x,
                tgt_max_num_tiles_y,  # not used, same as tgt_max_num_tiles_x
                tgt_n_tokens_per_tile,
                tgt_embed_dim,
            ) = self.global_token_positional_embedding.shape

            # resize ckpt to match instantiated shape
            inpt_global_pos_embed = self._resize_global_position_embedding(
                global_pos_embed=inpt_global_pos_embed,
                tgt_max_num_tiles=tgt_max_num_tiles_x,
                tgt_patch_grid_size=int(math.sqrt(tgt_n_tokens_per_tile - 1)),
            )

            if global_embed_is_sharded:
                inpt_global_pos_embed = distribute_tensor(
                    inpt_global_pos_embed,
                    device_mesh=global_embed_device_mesh,
                    placements=global_embed_placements,
                )

            # update state dict
            state_dict[prefix + "global_token_positional_embedding"] = inpt_global_pos_embed
            if inpt_global_pos_embed.shape != self.global_token_positional_embedding.shape:
                raise ValueError(
                    f"Loaded global positional embedding has shape {inpt_global_pos_embed.shape}, "
                    f"after interpolation. Expected shape {self.global_token_positional_embedding.shape}."
                )

    @staticmethod
    def _resize_local_position_embedding(local_pos_embed: torch.Tensor, tgt_patch_grid_size: int) -> torch.Tensor:
        """
        Interpolates the local position embedding for a vision encoder to accommodate
        a different number of tokens per tile. This is the only dimension that
        changes during interpolation.

        Args:
            local_pos_embed (torch.Tensor): The position embeddings tensor to be resized. It
                has shape [n_tokens_per_tile, emb_dim], where the first token is the CLS token
                and n_tokens_per_tile = patch_grid_size**2 + 1.
            tgt_patch_grid_size (int): The target size of each patch grid, i.e.,
                the square root of the number of tokens per tile, excluding the class token.

        Returns
        -------
            torch.Tensor: The resized position embeddings tensor of shape
                [tgt_n_tokens_per_tile, dim], where tgt_n_tokens_per_tile = tgt_patch_grid_size**2 + 1.

        Example:
            >>> import torch
            >>> import math
            >>> local_pos_embed = torch.randn((10*10+1, 64))  # Example input tensor
            >>> tgt_patch_grid_size = 20  # Target number of tokens per tile
            >>> resized_pos_embed = _resize_local_position_embedding(local_pos_embed, tgt_patch_grid_size)
            >>> print(resized_pos_embed.shape)
            torch.Size([20*20+1, 64])
        """
        # inverse n_tokens_per_tile = patch_grid_size**2 + 1, where +1 is the cls token
        inpt_n_tokens_per_tile, inpt_embed_dim = local_pos_embed.shape
        inpt_patch_grid_size = int(math.sqrt(inpt_n_tokens_per_tile - 1))

        # split tokens between cls and img tokens.
        # we don't want to interpolate cls token.
        cls_token, local_pos_embed = (
            local_pos_embed[[0]],  # cls token
            local_pos_embed[1:],  # image tokens
        )

        # we reshape n_tokens_per_tile - 1 --> (inpt_patch_grid_size, inpt_patch_grid_size)
        # and permute to have inpt_patch_grid_size as the last two dimensions
        # we also add a batch dim to the tensor, since F.interpolate expects it
        local_pos_embed = local_pos_embed.reshape(1, inpt_patch_grid_size, inpt_patch_grid_size, -1).permute(0, 3, 1, 2)

        local_pos_embed = F.interpolate(
            local_pos_embed,
            size=[tgt_patch_grid_size, tgt_patch_grid_size],
            mode="bilinear",
            align_corners=True,  # defaults from internal-llama-models
        )

        # reshape back to [1, tokens_per_tile, embed_dim]
        local_pos_embed = local_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, inpt_embed_dim)

        # remove batch dim added previously
        local_pos_embed = local_pos_embed.squeeze(0)

        # add cls token back in
        local_pos_embed = torch.cat([cls_token, local_pos_embed], dim=0)

        return local_pos_embed

    # TODO: Switch to public method after 2.5 is stable
    @staticmethod
    def _resize_global_position_embedding(
        global_pos_embed: torch.Tensor,
        tgt_max_num_tiles: int,
        tgt_patch_grid_size: int,
    ) -> torch.Tensor:
        """
        Interpolates the global position embedding for a vision encoder to accommodate new grid dimensions.
        The embedding dimension is not changed during interpolation, only max_num_tiles and num_tokens_per_tile.

        Args:
            global_pos_embed (torch.Tensor): The input global position embeddings tensor of shape
                [max_num_tiles_x, max_num_tiles_y, num_tokens_per_tile, embed_dim],
                where num_tokens_per_tile = inpt_patch_grid_size * inpt_patch_grid_size + 1 (CLS token), and
                max_num_tiles_x == max_num_tiles_y.
            tgt_max_num_tiles (int): The target maximum number of tiles along one dimension (assumed square grid).
            tgt_patch_grid_size (int): The target size of each patch grid, i.e., the square root of the number of tokens
                per tile, excluding the class token.


        Returns
        -------
            torch.Tensor: The resized global position embeddings tensor of shape
                [tgt_max_num_tiles, tgt_max_num_tiles, tgt_patch_grid_size * tgt_patch_grid_size + 1, embed_dim].

        Example:
            >>> import torch
            >>> global_pos_embed = torch.arange(3*3*(2*2+1)*4).reshape((3, 3, 2*2+1, 4))  # Example input tensor
            >>> tgt_max_num_tiles = 2  # Target maximum number of tiles
            >>> tgt_patch_grid_size = 3  # Target patch grid size
            >>> resized_global_pos_embed = (
            >>> _resize_global_position_embedding(global_pos_embed, tgt_max_num_tiles, tgt_patch_grid_size))
            >>> print(resized_global_pos_embed.shape)
            torch.Size([2, 2, 3*3+1, 4])
        """
        # remove cls token to interpolate it separately
        pos_embed = global_pos_embed[:, :, 1:, :]
        cls_embed = global_pos_embed[:, :, [0], :]

        (
            max_num_tiles_x,
            max_num_tiles_y,
            n_tokens_per_tile,
            embed_dim,
        ) = pos_embed.shape

        # tokens_per_tile == inpt_patch_grid_size**2
        # we reshape n_tokens_per_tile --> (inpt_patch_grid_size, inpt_patch_grid_size)
        inpt_patch_grid_size = int(math.sqrt(n_tokens_per_tile))
        pos_embed = pos_embed.reshape(
            max_num_tiles_x,
            max_num_tiles_y,
            inpt_patch_grid_size,
            inpt_patch_grid_size,
            embed_dim,
        )

        # combine max_num_tiles and patch_grid_size into one dimension
        pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
        pos_embed = pos_embed.reshape(
            max_num_tiles_x * inpt_patch_grid_size,
            max_num_tiles_y * inpt_patch_grid_size,
            embed_dim,
        )

        # add batch dim for interpolation
        pos_embed = pos_embed.unsqueeze(0)

        tgt_size = (
            int(tgt_max_num_tiles * tgt_patch_grid_size),
            int(tgt_max_num_tiles * tgt_patch_grid_size),
        )

        # move to the last two dim for interpolation
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            size=tgt_size,
            mode="bilinear",
            align_corners=True,  # defaults from internal-llama-models
        )

        # return to original shape and remove batch dim
        pos_embed = pos_embed.permute(0, 2, 3, 1).squeeze(0)

        # move it back in place
        pos_embed = pos_embed.view(
            tgt_max_num_tiles,
            tgt_patch_grid_size,
            tgt_max_num_tiles,
            tgt_patch_grid_size,
            embed_dim,
        )
        pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
        pos_embed = pos_embed.view(
            tgt_max_num_tiles,
            tgt_max_num_tiles,
            int(tgt_patch_grid_size**2),
            embed_dim,
        )

        # interpolate cls token
        cls_embed = cls_embed.permute(2, 3, 0, 1)
        cls_embed_resized = F.interpolate(
            cls_embed,
            size=(tgt_max_num_tiles, tgt_max_num_tiles),
            mode="bilinear",
            align_corners=True,  # defaults from internal-llama-models
        )
        cls_embed = cls_embed_resized.permute(2, 3, 0, 1)

        # add cls token back in
        global_pos_embed = torch.cat([cls_embed, pos_embed], dim=2)

        return global_pos_embed

    def forward(self, x: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape
                (bsz * n_imgs, n_tiles, n_tokens_per_tile, embed_dim).
            aspect_ratio (torch.Tensor): torch.Tensor with shape (bsz * n_imgs, 2),
                where aspect_ratio[k] represents the aspect ratio of the k^th image
                of the batch before tile-cropping,  e.g. aspect_ratio[k] = (2,1).

        Returns
        -------
            torch.Tensor: The input tensor with added positional embeddings.
        """
        bsz_and_n_imgs, n_tiles, n_tokens_per_tile, embed_dim = x.shape

        # apply local position embedding (same for every tile)
        x = x + (self.local_token_positional_embedding * (1 - self.gate.tanh()))

        # apply global positional embedding (different for every tile)
        x = x.view(bsz_and_n_imgs, n_tiles, n_tokens_per_tile, embed_dim)
        for batch_idx, (n_tiles_h, n_tiles_w) in enumerate(aspect_ratio):
            # When we batch images, all are padded to the same amount of tiles.
            # The aspect_ratio lets us know the non padded tiles for each image.
            # We only add positional encoding to those.
            n_non_padded_tiles = int(n_tiles_h * n_tiles_w)

            # We get only the positional encoding for non padded tiles,
            # i.e. n_tiles_h, n_tiles_w.
            pos_embed = self.global_token_positional_embedding[:n_tiles_h, :n_tiles_w, :, :]

            # Add pos encoding to the non padded tiles.
            pos_embed = pos_embed.reshape(n_non_padded_tiles, self.n_tokens_per_tile, embed_dim)
            pos_embed = pos_embed * self.gate.tanh()
            x[batch_idx, :n_non_padded_tiles, :, :] += pos_embed

        return x


class TilePositionalEmbedding(nn.Module):
    """
    Positional embedding for tiles, different for every tile, same for every token within a tile.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        embed_dim (int): The dimensionality of each tile embedding.
    """

    def __init__(
        self,
        max_num_tiles: int,
        embed_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        scale = embed_dim**-0.5
        self.embedding = nn.Parameter(scale * torch.randn(max_num_tiles, max_num_tiles, 1, embed_dim))
        self.gate = nn.Parameter(torch.zeros(1))

        # Register load hook to interpolate positional embeddings
        self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

    # TODO: Switch to public method after 2.5 is stable
    @torch.no_grad()
    def _load_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Tuple[Any],
        **kwargs: Dict[str, Any],
    ):
        """
        Interpolates positional embeddings to accomodate different number of tiles,
        in case the model was instantiated with different
        settings than the one you are loading the state dict from.

        For more info, check self._dynamic_resize function.

        Args:
            state_dict (Dict[str, Any]): The state dict to load.
            prefix (str): The prefix of the state dict.
            *args (Tuple[Any]): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Raises
        ------
            ValueError: if the shape of the loaded embedding is not compatible with the current embedding.
            ValueError: if max_num_tiles_x, max_num_tiles_y are not equal.
            ValueError: if after interpolation, the shape of the loaded embedding is not compatible with the current embedding.
        """
        embedding = state_dict.get(prefix + "embedding")

        if embedding is not None:
            # We can only apply F.interpolate to vanilla tensors, not DTensors
            # If pos embeds are a DTensor, we gather the full tensor, apply
            # interpolate, and then reshard after
            if isinstance(embedding, DTensor):
                embedding_is_sharded = True
                device_mesh = embedding.device_mesh
                placements = embedding.placements
                embedding = embedding.full_tensor()
            else:
                embedding_is_sharded = False

            # ckpt pos emb
            (
                tgt_max_num_tiles_x,
                tgt_max_num_tiles_y,
                tgt_num_tokens,
                tgt_emb,
            ) = self.embedding.shape

            # instantiated pos emb
            (
                inpt_max_num_tiles_x,
                inpt_max_num_tiles_y,
                inpt_num_tokens,
                inpt_emb,
            ) = state_dict[prefix + "embedding"].shape

            # sanity check
            if inpt_num_tokens != tgt_num_tokens or inpt_emb != tgt_emb:
                raise ValueError(
                    "Expected embedding shape to be (..., num_tokens, tgt_emb) to match"
                    f" but found shapes {self.embedding.shape} and {state_dict[prefix + 'embedding'].shape}"
                )

            if inpt_max_num_tiles_x != inpt_max_num_tiles_y:
                raise ValueError(
                    "Expected max_num_tiles_x, max_num_tiles_y to be equal but found, but found"
                    f"(max_num_tiles_x, max_num_tiles_y, 1, embed_dim) = {self.embedding.shape}"
                )

            # resize ckpt to match instantiated shape
            embedding_new = self._resize_position_embedding(embedding, tgt_max_num_tiles=tgt_max_num_tiles_x)

            if embedding_is_sharded:
                embedding_new = distribute_tensor(
                    embedding_new,
                    device_mesh=device_mesh,
                    placements=placements,
                )

            # update state dict
            state_dict[prefix + "embedding"] = embedding_new
            if embedding_new.shape != self.embedding.shape:
                raise ValueError(
                    "Expected embedding shape and embedding_new.shape to match"
                    f" but found shapes {self.embedding.shape} and {embedding_new.shape}"
                )

    @staticmethod
    def _resize_position_embedding(embedding: torch.Tensor, tgt_max_num_tiles: int) -> torch.Tensor:
        """
        Interpolates positional embeddings to accomodate a different max_num_tiles. These
        are the only dimensions that changes during interpolation.

        Args:
            embedding (torch.Tensor): torch.Tensor with shape (max_num_tiles, max_num_tiles, 1, embed_dim
            tgt_max_num_tiles (int): The number of tiles to resize to.

        Returns
        -------
            torch.Tensor: The resized embedding.

        Example:
            >>> import torch
            >>> # create dummy embedding
            >>> embedding = torch.arange(2*2*2*2).reshape(2, 2, 2, 2).float()
            >>> resized_embed = _dynamic_resize(embedding, tgt_max_num_tiles=1)
            >>> print(resized_embed.shape)
            >>> torch.Size([1, 1, 2, 2])
        """
        # set max_num_tiles to the last dimension
        embedding = embedding.permute(2, 3, 0, 1)

        embedding = F.interpolate(
            embedding,
            size=(tgt_max_num_tiles, tgt_max_num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        # permute to the original shape
        embedding = embedding.permute(2, 3, 0, 1)
        return embedding

    def forward(self, x: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:
        """
        args:
            x (torch.Tensor): torch.Tensor with shape (bsz * n_imgs, n_tiles, n_tokens_per_tile, embed_dim).
            aspect_ratio (torch.Tensor): torch.Tensor with shape (bsz * n_imgs, 2),
                representing the aspect ratio of the image before tile-cropping, e.g. (2,1).

        Returns
        -------
            torch.Tensor: The input tensor with added positional embeddings.
        """
        for batch_idx, (n_tiles_h, n_tiles_w) in enumerate(aspect_ratio):
            # When we batch images, all are padded to the same amount of tiles.
            # The aspect_ratio lets us know the non padded tiles for each image.
            # We only add positional encoding to those.
            n_non_padded_tiles = int(n_tiles_h * n_tiles_w)

            # We get only the positional encoding for non padded tiles,
            # i.e. n_tiles_h, n_tiles_w.
            pos_embed = self.embedding[:n_tiles_h, :n_tiles_w, :, :]

            # Add pos encoding to the non padded tiles.
            pos_embed = pos_embed.reshape(n_non_padded_tiles, 1, self.embed_dim)
            x[batch_idx, :n_non_padded_tiles, :, :] += pos_embed * self.gate.tanh()

        return x


class VisionTransformer(nn.Module):
    """
    Implementation of the ViT architecture (https://arxiv.org/abs/2010.11929),
    with support for tile-cropped images, outputting of hidden layers and optional CLS projection.

    ViT is a transformer architecture that takes in images and outputs N embedded tokens that
    represent this image. Each image is divided into **patches** by a convolution.
    These patches are flattened and subsequently treated as **tokens** by the transformer.

    To further enhance the performance of ViT and avoid downscaling images, we support tile-cropped images,
    which are images divided into **tiles** during the preprocessing stage. For example, instead of
    downscaling an 800x400 image to fit 400x400, we may crop it into two 400x400 tiles,
    if the ``tile_size=400``. For details on preprocessing, please refer to
    :class:`torchtune.models.clip._transforms.CLIPImageTransform`.

    Each of these tiles is further broken down into patches by a convolution operation. For example, if
    your ``patch_size=40``, then each (400, 400) tile will become a grid of 10x10 patches, and your whole image will have
    num_tiles * n_tokens -> num_tiles * (10x10 patches + 1 CLS token) -> num_tiles * 101.

    Before the transformer layers, a CLS token is added to each tile as the first token.
    In transformers, a token called CLS is a special token that is added to the beginning of each sequence.
    This token can be used to represent the whole input, instead of using a pooling operation, for example.

    To help the model "see" the whole image, we use positional embeddings. If your image
    was tile-cropped, then you need to use tile positional embeddings:

    - token_pos_embedding (tiled): :class:`torchtune.models.clip._position_embeddings.TiledTokenPositionalEmbedding`
    - pre_tile_pos_embed: :class:`torchtune.models.clip._position_embeddings.TilePositionalEmbedding`
    - post_tile_pos_embed: :class:`torchtune.models.clip._position_embeddings.TilePositionalEmbedding`

    Otherwise, pre and post tile_pos_embed should be None and all you need is a simple
    token positional embedding:

    - token_pos_embedding (not tiled): :class:`torchtune.models.clip._position_embeddings.TokenPositionalEmbedding`

    All images will be considered as a stack of tiles, even if your image was not tile-cropped. In such cases,
    your image would be composed of a single tile.

    In summary:

    1) An image is broken down into tiles during preprocessing.
    2) In the ViT, the tiles will be broken down into patches.
    3) The patches will be flattened and transformed. We call them tokens, because that's how the transformer sees them.


    Image: shape (8x8)

    .. code-block:: text

        |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |
        |  9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
        | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
        | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
        | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 |
        | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 |
        | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 |
        | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 |

    Tiles: shape (4,4,4) # (num_tiles, tile_size, tile_size)

    .. code-block:: text

        |  1 |  2 |  3 |  4 |    |  5 |  6 |  7 |  8 |
        |  9 | 10 | 11 | 12 |    | 13 | 14 | 15 | 16 |
        | 17 | 18 | 19 | 20 |    | 21 | 22 | 23 | 24 |
        | 25 | 26 | 27 | 28 |    | 29 | 30 | 31 | 32 |

        | 33 | 34 | 35 | 36 |    | 37 | 38 | 39 | 40 |
        | 41 | 42 | 43 | 44 |    | 45 | 46 | 47 | 48 |
        | 49 | 50 | 51 | 52 |    | 53 | 54 | 55 | 56 |
        | 57 | 58 | 59 | 60 |    | 61 | 62 | 63 | 64 |

    Patches: shape (4,4,2,2) # (num_tiles, num_patches_per_tile, patch_size, patch_size)

    .. code-block:: text

        |  1 |  2 |    |  3 |  4 |    |  5 |  6 |    |  7 |  8 |
        |  9 | 10 |    | 11 | 12 |    | 13 | 14 |    | 15 | 16 |

        | 17 | 18 |    | 19 | 20 |    | 21 | 22 |    | 23 | 24 |
        | 25 | 26 |    | 27 | 28 |    | 29 | 30 |    | 31 | 32 |

        | 33 | 34 |    | 35 | 36 |    | 37 | 38 |    | 39 | 40 |
        | 41 | 42 |    | 43 | 44 |    | 45 | 46 |    | 47 | 48 |

        | 49 | 50 |    | 51 | 52 |    | 53 | 54 |    | 55 | 56 |
        | 57 | 58 |    | 59 | 60 |    | 61 | 62 |    | 63 | 64 |

    token: shape (4, 4, 4) # (num_tiles, num_patches_per_tile, emb_dim)

    .. code-block:: text

        |  1 |  2 |  9 |  10 |    |  3 |  4 |  11 |  12 |    |  17 |  18 |  25 |  26 |    | 19 | 20 |  27 |  28 |
        | ... continuation of data ...
        | ... continuation of data ...
        | 37 | 38 | 45 |  46 |    | 39 |  40 | 47 |  48 |    | 53 | 54 |  61 |  62 |    | 55 | 56 |  63 |  64 |

    For the positional embeddings:

    Same for every tile, different for every token.

    - :class:`torchtune.models.clip._position_embeddings.TokenPositionalEmbedding`
    - :class:`torchtune.models.clip._position_embeddings.TiledTokenPositionalEmbedding`

    .. code-block:: text

        |  1 |  2 |  3 |  4 |    |  1 |  2 |  3 |  4 |
        |  9 | 10 | 11 | 12 |    |  9 | 10 | 11 | 12 |
        | 17 | 18 | 19 | 20 |    | 17 | 18 | 19 | 20 |
        | 25 | 26 | 27 | 28 |    | 25 | 26 | 27 | 28 |

        |  1 |  2 |  3 |  4 |    |  1 |  2 |  3 |  4 |
        |  9 | 10 | 11 | 12 |    |  9 | 10 | 11 | 12 |
        | 17 | 18 | 19 | 20 |    | 17 | 18 | 19 | 20 |
        | 25 | 26 | 27 | 28 |    | 25 | 26 | 27 | 28 |

    Different for every tile, different for every token.

    - :class:`torchtune.models.clip._position_embeddings.TiledTokenPositionalEmbedding`

    .. code-block:: text

        |  1 |  2 |    |  3 |  4 |    |  5 |  6 |    |  7 |  8 |
        |  9 | 10 |    | 11 | 12 |    | 13 | 14 |    | 15 | 16 |

        | 17 | 18 |    | 19 | 20 |    | 21 | 22 |    | 23 | 24 |
        | 25 | 26 |    | 27 | 28 |    | 29 | 30 |    | 31 | 32 |

        | 33 | 34 |    | 35 | 36 |    | 37 | 38 |    | 39 | 40 |
        | 41 | 42 |    | 43 | 44 |    | 45 | 46 |    | 47 | 48 |

        | 49 | 50 |    | 51 | 52 |    | 53 | 54 |    | 55 | 56 |
        | 57 | 58 |    | 59 | 60 |    | 61 | 62 |    | 63 | 64 |

    different for every tile, same for every token within a tile.

    - :class:`torchtune.models.clip._position_embeddings.TilePositionalEmbedding`

    .. code-block:: text

        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |

        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |

    Args:
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
            with shape (40, 40) each.
        num_layers (int): The number of transformer layers.
        embed_dim (int): The dimensionality of each patch embedding (token).
        layer (nn.Module): The transformer layer module.
        token_pos_embedding (nn.Module): The token positional embedding module.
        pre_tile_pos_embed (Optional[nn.Module]): The pre-tile positional embedding module. It should be
            None if your image was not tile-cropped in advance.
        post_tile_pos_embed (Optional[nn.Module]): The post-tile positional embedding module. It should be
            None if your image was not tile-cropped in advance.
        cls_projection (Optional[nn.Module]): The CLS projection module. It should take an input tensor
            of shape (bsz * n_tiles, n_tokens, embed_dim) and output a tensor of shape
            (bsz * n_tiles, cls_output_dim). If provided, only the CLS token projection will be
            outputted, instead of all tokens.
        out_indices (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, ``out_indices=[0,3]`` will
            return the tokens before they go through the first and fourth layers.
        in_channels (int): The number of image input channels.

    Raises
    ------
        ValueError: If `tile_size` is not greater than 0.
        ValueError: If `patch_size` is not greater than 0.
        ValueError: If `len(out_indices)` is greater than `num_layers`.
    """

    def __init__(
        self,
        patch_size: int,
        tile_size: int,
        num_layers: int,
        embed_dim: int,
        layer: nn.Module,
        token_pos_embedding: nn.Module,
        pre_tile_pos_embed: Optional[nn.Module] = None,
        post_tile_pos_embed: Optional[nn.Module] = None,
        cls_projection: Optional[nn.Module] = None,
        out_indices: Optional[List[int]] = None,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if out_indices and (len(out_indices) > num_layers):
            raise ValueError(f"len(out_indices) must be <= num_layers. Got {out_indices=} and {num_layers=}")

        # constants
        patch_grid_size = tile_size // patch_size
        self.patches_per_tile = patch_grid_size**2
        self.out_indices = out_indices
        if not out_indices:
            self.out_indices = []

        # input modules
        self.pre_tile_pos_embed = pre_tile_pos_embed
        self.post_tile_pos_embed = post_tile_pos_embed
        self.token_pos_embedding = token_pos_embedding

        self.cls_projection = cls_projection
        self.layers = _get_clones(layer, num_layers)

        # other modules
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=False,
        )

        self.ln_post = Fp32LayerNorm(embed_dim)
        self.ln_pre = Fp32LayerNorm(embed_dim)

        self.cls_token_embedding = CLSEmbedding(embed_dim)

    def get_image_tokens_per_tile(self):
        return self.patches_per_tile + 1  # +1 for CLS token

    def forward(
        self,
        images: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Processes images and returns the tokens and hidden states.

        Multiple images per sample: we add a dimension n_imgs to the input. This is useful when a single
        sample constains multiple images, for example:

        - sample 1: "<image> what animal is this?"
        - sample 2: "I like <image> more than <image>"

        In this case, sample 1 has one image, and sample 2 has two images. max_n_imgs = max(2,1) = 2.
        So your input should have shape (bsz=2, n_imgs=2, num_tiles, n_channels, tile_size, tile_size).

        Notice that to batch it, you will have to pad n_imgs to max_n_imgs and max_num_tiles.

        Args:
            images (torch.Tensor): torch.Tensor with shape (bsz, n_imgs, n_tiles, n_channels, tile_size, tile_size).
            aspect_ratio (Optional[torch.Tensor]): torch.Tensor with shape (bsz, n_imgs, 2). If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.

        Returns
        -------
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple: (x, hidden_states),
                where x is a torch.tensor of shape (bsz, n_imgs, n_tiles, n_tokens, embed_dim) and
                hidden_states has shape is a list of len(out_indices) torch.tensor with shape
                (bsz, n_imgs, n_tiles, n_tokens, embed_dim).

        Raises
        ------
            ValueError: If aspect_ratio is None, but n_tiles > 1 in the batch.

        Examples
        --------
            >>> from torchtune.modules.transforms.vision_utils.tile_crop import tile_crop
            >>> from torchtune.modules import VisionTransformer
            >>>
            >>> num_channels = 3
            >>> image_size = (800,400)
            >>> tile_size = 400
            >>> patch_size=40
            >>> patch_grid_size = tile_size // patch_size
            >>>
            >>> # for details about preprocessing, please check
            >>> # torchtune.models.clip._transforms.CLIPImageTransform
            >>>
            >>> # create a random image
            >>> image = torch.rand(num_channels, image_size[0], image_size[1])
            >>>
            >>> # (num_tiles, nch, h, w) -> (2, 3, 400, 400)
            >>> tile_cropped_image = tile_crop(image, tile_size)
            >>> aspect_ratio = torch.tensor([2,1])
            >>>
            >>> # make it a batch of 1 image
            >>> batch_image = tile_cropped_image.unsqueeze(0)
            >>> batch_aspect_ratio = aspect_ratio.unsqueeze(0)
            >>>
            >>> # make it have only 1 image per sample
            >>> batch_image = tile_cropped_image.unsqueeze(1)
            >>> batch_aspect_ratio = aspect_ratio.unsqueeze(1)
            >>>
            >>> # For a detailed example, please check
            >>> # torchtune.models.clip._position_embeddings.clip_vision_encoder
            >>> # model = VisionTransformer(
            ... #           out_indices = [1,2,3,4,5],
            ... #           patch_size=40,
            ... #           patch_grid_size = patch_grid_size,
            ... #           embed_dim = 32,
            ... #           num_layers = 6,
            ... #           in_channels = num_channels,
            ... #           ...)
            >>>
            >>> x, hidden_states = model(images = batch_image, aspect_ratio = batch_aspect_ratio)
            >>>
            >>> # (bsz, n_imgs, num_tiles, num_patches_per_tile + CLS token, embed_dim)
            >>> print(x.shape)
            torch.Size([1, 1, 2, 101, 32])
            >>>
            >>> # list with tensors of shape (bsz, n_imgs, num_tiles, num_patches_per_tile + CLS token, embed_dim)
            >>> print(len(hidden_states))
            5
        """
        hidden_states = []

        # parse inputs
        bsz, n_imgs, n_tiles, nch, w, h = images.shape
        bsz_and_n_imgs = bsz * n_imgs

        # if aspect_ratio is not provided, it defaults to one tile [1,1]
        if aspect_ratio is None:
            aspect_ratio = torch.ones((bsz_and_n_imgs, 2), dtype=torch.int, device=images.device)
            if n_tiles > 1:
                raise ValueError(
                    f"aspect_ratio was not provided, but found n_tiles>1 for {images.shape=}. Please provide aspect_ratio."
                )

        images = images.reshape(bsz_and_n_imgs * n_tiles, nch, w, h)
        aspect_ratio = aspect_ratio.reshape(bsz_and_n_imgs, 2)

        # patch embeddings (tokens)
        # A tile becomes a grid of patch_grid_size X patch_grid_size patches
        # these patches are flatenned, and called tokens from here on.

        # out: (bsz * n_imgs * n_tiles, embed_dim, patch_grid_size, patch_grid_size)
        x = self.conv(images)

        # out: (bsz * n_imgs, n_tiles, n_tokens, embed_dim)
        x = x.reshape(bsz_and_n_imgs, n_tiles, -1, self.patches_per_tile).permute(0, 1, 3, 2)
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape

        # pre_tile_pos_embed
        if self.pre_tile_pos_embed:
            x = self.pre_tile_pos_embed(x, aspect_ratio)

        # insert cls token
        x = self.cls_token_embedding(x)
        n_tokens += 1

        # token_pos_embedding
        x = self.token_pos_embedding(x, aspect_ratio)

        # norm
        x = self.ln_pre(x)

        # transformer with optional hidden layer outputs
        x = x.reshape(bsz_and_n_imgs, n_tiles * n_tokens, embed_dim)
        for layer_idx, transformer_layer in enumerate(self.layers):
            if layer_idx in self.out_indices:
                h = x.reshape(bsz, n_imgs, n_tiles, n_tokens, embed_dim)
                hidden_states.append(h)
            x = transformer_layer(x)

        # norm
        x = self.ln_post(x)

        # post_tile_pos_embed
        if self.post_tile_pos_embed:
            x = x.reshape(bsz_and_n_imgs, n_tiles, n_tokens, embed_dim)
            x = self.post_tile_pos_embed(x, aspect_ratio)

        # reshape output
        x = x.reshape(bsz, n_imgs, n_tiles, n_tokens, embed_dim)

        # cls token projection. n_tokens becomes 1
        if self.cls_projection:
            x = self.cls_projection(x)

        return x, hidden_states


class CLSEmbedding(nn.Module):
    """
    Adds a CLS token to every tile in an image.

    Notice that tile is different from patch (token). An image is divided into tiles during pre-processing,
    and patches are the outcome of the convolution in the ViT applied to each tile.

    Args:
        embed_dim (int): The dimensionality of the input patch embedding.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        scale = embed_dim**-0.5
        self.weight = nn.Parameter(scale * torch.randn(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add 1 CLS token to every tile
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape
        cls_emb = self.weight.broadcast_to(bsz_and_n_imgs, n_tiles, 1, embed_dim)
        return torch.cat([cls_emb, x], dim=2)


class CLSProjection(nn.Module):
    """
    Linear projection of the CLS token.

    Args:
        embed_dim (int): The dimensionality of the input patch embedding.
        cls_output_dim (int): The dimensionality of the output projection.
    """

    def __init__(self, embed_dim: int, cls_output_dim: int) -> None:
        super().__init__()

        scale = embed_dim**-0.5
        self.cls_output_dim = cls_output_dim
        self.weight = nn.Parameter(scale * torch.randn(embed_dim, cls_output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_imgs, n_tiles, n_tokens, embed_dim = x.shape
        x = x.reshape(bsz * n_imgs * n_tiles, n_tokens, embed_dim)

        # out: (bsz * n_tiles, cls_output_dim)
        x = x[:, 0, :] @ self.weight

        # num_tokens becomes 1 because we only return the CLS token projection
        x = x.reshape(bsz, n_imgs, n_tiles, 1, self.cls_output_dim)
        return x


def clip_vision_encoder(
    tile_size: int,
    patch_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    activation: Callable = nn.SiLU,
    cls_output_dim: int = 512,
    attn_bias: bool = True,
    out_indices: Optional[List[int]] = None,
    output_cls_projection: bool = False,
    max_num_tiles: int = 4,
    in_channels: int = 3,
    intermediate_act: torch.nn.Module = torch.nn.SiLU(),
) -> VisionTransformer:
    """
    Builds the vision encoder associated with the clip model. This includes:

    - TransformerEncoderLayer
    - positional embeddings
    - CLS projection (optional)

    For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        embed_dim (int): The dimensionality of each patch embedding (token).
        num_layers (int): The number of transformer layers.
        num_heads (int): The number of attention heads in each transformer layer.
        activation (Callable): The activation function to use in the MLP layer.
        cls_output_dim (int): The dimensionality of the output tensor from the CLS projection module.
        attn_bias (bool): Boolean for if to use bias in the attention module. Default True.
        out_indices (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, ``out_indices=[0,3]`` will
            return the tokens before they go through the first and fourth layers.
        output_cls_projection (bool): If True, only the CLS token projection will be outputted,
            instead of all tokens. Defaults to False.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings.
        in_channels (int): The number of image input channels.
        intermediate_act (torch.nn.Module): The activation function used in the intermediate layers in the transformer encoder.

    Returns
    -------
        A `VisionTransformer` object.

    Raises
    ------
        AssertionError: If ``embed_dim`` is not divisible by ``num_heads``.
    """
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    cls_projection = (
        CLSProjection(embed_dim=embed_dim, cls_output_dim=cls_output_dim) if output_cls_projection else None
    )

    # transformer layer
    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=embed_dim // num_heads,
        q_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        k_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        v_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        pos_embeddings=None,
        attn_dropout=0.0,
        is_causal=False,
    )
    mlp = clip_mlp(
        in_dim=embed_dim,
        hidden_dim=4 * embed_dim,
        out_dim=embed_dim,
        activation=activation(),
    )
    transformer_layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=Fp32LayerNorm(embed_dim, eps=1e-5),
        mlp_norm=Fp32LayerNorm(embed_dim, eps=1e-5),
        sa_scale=None,
        mlp_scale=None,
    )

    # position embeddings
    if max_num_tiles == 1:
        pre_tile_pos_embed = None
        post_tile_pos_embed = None
        token_pos_embedding = TokenPositionalEmbedding(embed_dim=embed_dim, patch_size=patch_size, tile_size=tile_size)
    else:
        pre_tile_pos_embed = TilePositionalEmbedding(max_num_tiles=max_num_tiles, embed_dim=embed_dim)
        post_tile_pos_embed = TilePositionalEmbedding(max_num_tiles=max_num_tiles, embed_dim=embed_dim)
        token_pos_embedding = TiledTokenPositionalEmbedding(
            max_num_tiles=max_num_tiles,
            embed_dim=embed_dim,
            patch_size=patch_size,
            tile_size=tile_size,
        )

    return VisionTransformer(
        num_layers=num_layers,
        layer=transformer_layer,
        token_pos_embedding=token_pos_embedding,
        pre_tile_pos_embed=pre_tile_pos_embed,
        post_tile_pos_embed=post_tile_pos_embed,
        cls_projection=cls_projection,
        out_indices=out_indices,
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
    )


def clip_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    activation: nn.Module,
    quantize_base: bool = False,
) -> FeedForward:
    """
    Build the MLP layer associated with the clip model.
    """
    gate_proj = nn.Linear(in_dim, hidden_dim) if not quantize_base else FrozenNF4Linear(in_dim, hidden_dim, bias=True)
    down_proj = nn.Linear(hidden_dim, out_dim) if not quantize_base else FrozenNF4Linear(hidden_dim, out_dim, bias=True)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=None, activation=activation)


## attention


def torch_version_ge(version: str) -> bool:
    """
    Check if torch version is greater than or equal to the given version.

    Args:
        version (str): The torch version to compare against

    Returns
    -------
        bool: True if torch version is greater than or equal to the given version.

    Example:
        >>> print(torch.__version__)
        2.4.0
        >>> torch_version_ge("2.0")
        True
    """
    return version in torch.__version__ or torch.__version__ >= version


def log_rank_zero(logger: logging.Logger, msg: str, level: int = logging.INFO) -> None:
    """
    Logs a message only on rank zero.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    logger.log(level, msg)


@lru_cache(None)
def log_once(logger: logging.Logger, msg: str, level: int = logging.INFO) -> None:
    """
    Logs a message only once. LRU cache is used to ensure a specific message is
    logged only once, similar to how :func:`~warnings.warn` works when the ``once``
    rule is set via command-line or environment variable.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    log_rank_zero(logger=logger, msg=msg, level=level)


# We can only use flex attention / BlockMask if torch version >= 2.5.0 and GPU is Turing / SM75 and above
_SUPPORTS_FLEX_ATTENTION = (
    torch_version_ge("2.5.0") and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5)
)

if _SUPPORTS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        flex_attention,
    )

    flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

    # We cannot do nested compile, but flex attention only has perf benefits
    # when compiled. To insulate it from the compiler, we wrap it with
    # compiler.disable so that it can be used regardless of whether the model
    # is compiled or not, and flex attention always remains compiled.
    @torch.compiler.disable(recursive=False)
    def compile_friendly_flex_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        return flex_attention_compiled(q, k, v, block_mask=block_mask)

    _MaskType = Union[torch.Tensor, BlockMask]
else:
    _MaskType = torch.Tensor


def _sdpa_or_flex_attention() -> Callable:
    """
    Helper function to decide when to call flex attention or SDPA. It will use
    flex attention if ALL of the following conditions are met, otherwise it will
    default to SDPA:
    - torch version >= 2.5.0
    - we are sample packing, therefore mask is a BlockMask
    - torch.cuda.get_device_capability() >= (7, 5)
    """
    if _SUPPORTS_FLEX_ATTENTION:

        def _attention_call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[_MaskType],
            dropout_p: float,
            is_causal: bool,
        ) -> torch.Tensor:
            # Flex attention uses the BlockMask
            # (https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py#L168)
            # instead of a traditional boolean tensor mask. If this is passed in,
            # we assume the user wants to use flex attention instead of traditional SDPA.
            # This will use flash attention under the hood with support for custom masks.
            # Currently, it is used when sample packing is enabled (see torchtune.datasets.PackedDataset)
            if isinstance(mask, BlockMask):
                log_once(
                    logger,
                    "Using flex attention for attention computation since a BlockMask was passed in.",
                    level=logging.DEBUG,
                )
                if dropout_p > 0.0:
                    raise ValueError("Flex attention does not support dropout. Please set dropout to 0.0.")
                return compile_friendly_flex_attention(
                    q,
                    k,
                    v,
                    block_mask=mask,
                )
            # If mask is a standard boolean tensor or None, then use SDPA
            else:
                # shape: [b, 1, s, s]
                if mask is not None:
                    mask = mask[:, None, :, :]

                # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
                return nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )

    else:

        def _attention_call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[_MaskType],
            dropout_p: float,
            is_causal: bool,
        ) -> torch.Tensor:
            # shape: [b, 1, s, s]
            if mask is not None:
                mask = mask[:, None, :, :]

            # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
            return nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    return _attention_call


class KVCache(nn.Module):
    """
    Standalone ``nn.Module`` containing a kv-cache to cache past key and values during inference.

    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_heads (int): number of heads. We take num_heads instead of num_kv_heads because
            the cache is created after we've expanded the key and value tensors to have the
            same shape as the query tensor. See attention.py for more details
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("cache_pos", torch.arange(0, cache_shape[2]), persistent=False)
        self.batch_size = batch_size

    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos -= self.size

    @property
    def size(self) -> int:
        return self.cache_pos[0].item()

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with the new ``k_val``, ``v_val`` and return the updated cache.

        Note:
            When updating the KV cache, it is assumed that subsequent updates should update key-value
            positions in consecutive sequence positions. If you wish to update cache values which have
            already been filled, use ``.reset()``, which will reset the cache to the zero-th position.

        Example:
            >>> cache = KVCache(batch_size=2, max_seq_len=16, num_heads=4, head_dim=32, dtype=torch.bfloat16)
            >>> keys, values = torch.ones((2, 4, 8, 32)), torch.ones((2, 4, 8, 32))
            >>> cache.update(keys, values)
            >>> # now positions 0 through 7 are filled
            >>> cache.size
            >>> 8
            >>> keys, values = torch.ones((2, 4, 1, 32)), torch.ones((2, 4, 1, 32))
            >>> cache.update(keys, values)
            >>> # this will fill at position 8
            >>> cache.size
            >>> 9

        Args:
            k_val (torch.Tensor): Current key tensor with shape [B, H, S, D]
            v_val (torch.Tensor): Current value tensor with shape [B, H, S, D]

        Returns
        -------
            Tuple[torch.Tensor, torch.Tensor]: Updated key and value cache tensors, respectively.

        Raises
        ------
            AssertionError: if the sequence length of ``k_val`` is longer than the maximum cache sequence length.
            ValueError: if the batch size of the new key (or value) tensor is greater than the batch size
                used during cache setup.
        """
        bsz, _, seq_len, _ = k_val.shape
        if bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"The current cache has been setup with a batch size of {self.k_cache.shape[0]}"
                f", but found new key tensors with batch size {k_val.shape[0]}!"
            )

        assert (self.cache_pos[0] + seq_len) <= self.k_cache.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:, :, self.cache_pos[:seq_len]] = k_val
        v_out[:, :, self.cache_pos[:seq_len]] = v_val

        # forward cache_pos seq_len positions along
        # cache_pos starts at (0, 1, 2, 3, 4, 5, ...)
        # an update of seq_len = 5 tokens brings it to
        # (5, 6, 7, 8, 9, ...)
        # this allows us to track the current position in the cache
        # after the last update in a compile-friendly way without any dynamism
        # e.g. relying on an int size tracker, or re-creating cache_pos every time
        self.cache_pos += seq_len

        return k_out, v_out


class MultiHeadAttention(nn.Module):
    """Multi-headed attention layer with support for grouped query
    attention (GQA) introduced in https://arxiv.org/abs/2305.13245v1.

    GQA is a version of multiheaded attention (MHA) which uses fewer
    key/value heads than query heads by grouping n query heads for each
    key and value head. Multi-Query Attention is an extreme
    version where we have a single key and value head shared by all
    query heads.

    Following is an example of MHA, GQA and MQA with num_heads = 4

    (credit for the documentation:
    `litgpt.Config <https://github.com/Lightning-AI/litgpt/blob/eda1aaaf391fd689664f95487ab03dc137e213fd/litgpt/config.py>`_).


    ::

        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │         │        │                 │
        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
                MHA                    GQA                   MQA
        n_kv_heads =4          n_kv_heads=2           n_kv_heads=1

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            ``num_heads % num_kv_heads == 0``. For standard MHA set ``num_kv_heads == num_heads``,
            for GQA ``num_kv_heads < num_heads``, and for MQA set ``num_kv_heads == 1``.
        head_dim (int): dimension of each head, calculated by ``embed_dim // num_heads``.
        q_proj (nn.Module): projection layer for query.
        k_proj (nn.Module): projection layer for key.
        v_proj (nn.Module): projection layer for value.
        output_proj (nn.Module): projection layer for output.
        pos_embeddings (Optional[nn.Module]): positional embeddings layer, e.g. RotaryPositionalEmbeddings.
        q_norm (Optional[nn.Module]): normalization layer for query, e.g. RMSNorm. For decoding, this is applied
            before updating from kv_cache. This means it will only support token wide normalization and not
            batch or sequence wide normalization.
        k_norm (Optional[nn.Module]): normalization layer for key, must be set if q_norm is.
        kv_cache (Optional[KVCache]): KVCache object used to cache key and value
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default: 4096.
        is_causal (bool): sets the default mask to causal when no mask is provided
        attn_dropout (float): dropout value passed onto the scaled_dot_product_attention function.
            Default value is 0.0.

    Raises
    ------
        ValueError: If ``num_heads % num_kv_heads != 0``
        ValueError: If ``embed_dim % num_heads != 0``
        ValueError: If ``attn_dropout < 0`` or ``attn_dropout > 1``
        ValueError: if q_norm is defined without k_norm or vice versa
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: Optional[nn.Module] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by " f"num_kv_heads ({num_kv_heads})")

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by " f"num_heads ({num_heads})")

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q and k norm must be set together")

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        # Set layers
        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.pos_embeddings = pos_embeddings

        # Use flex attention if supported and we are sample packing
        self._attention_call = _sdpa_or_flex_attention()

        # this flag indicates whether to update the kv-cache during forward
        # passes. when disabled, we can have the cache setup but still
        # perform normal forward passes
        self.cache_enabled = False

    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: int) -> None:
        """Setup key value caches for attention calculation. If called
        after kv_cache is already setup, this will be skipped.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            max_seq_len (int): maximum sequence length model will be run with.
        """
        # Don't overwrite user defined kv_cache from init
        if self.kv_cache is not None:
            logger.warning("Key value caches are already setup. You cannot call ``setup_caches()`` twice. Skipping.")
        else:
            self.kv_cache = KVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )
            self.cache_enabled = True

    def reset_cache(self):
        """Reset the key value caches."""
        if self.kv_cache is None:
            raise RuntimeError("Key value caches are not setup. Call ``setup_caches()`` first.")
        self.kv_cache.reset()

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x s_x x d] for the query
            y (Optional[torch.Tensor]): second input tensor with shape [b x s_y x d], is the input
                for k and v. For self attention, x=y. Optional only with kv_cache enabled.
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Raises
        ------
            ValueError: If no ``y`` input and ``kv_cache`` is not enabled.

        Returns
        -------
            torch.Tensor: output tensor with attention applied

        Notation used for tensor shapes:
            - b: batch size
            - s_x: sequence length for x
            - s_y: sequence length for y
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim
        """
        # x has shape [b, s_x, d]
        # y has shape [b, s_y, d]
        b, s_x, _ = x.shape
        s_y = y.shape[1] if y is not None else 0

        # q has shape [b, s_x, num_heads * head_dim]
        q = self.q_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # Apply positional embeddings
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)

        # [b, n_h, s_x, h_d]
        q = q.transpose(1, 2)

        # Normalize q
        if self.q_norm is not None:
            q = self.q_norm(q)

        if y is None:
            if self.kv_cache is None:
                raise ValueError("Must provide y input or use kv_cache to enable streaming decoding")
            k = self.kv_cache.k_cache
            v = self.kv_cache.v_cache
        else:
            # Update k and v shape, positional embeddings, and normalization

            # k has shape [b, s_y, num_kv_heads * head_dim]
            # v has shape [b, s_y, num_kv_heads * head_dim]
            k = self.k_proj(y)
            v = self.v_proj(y)

            # Apply positional embeddings
            # k: [b, s_y, n_kv, h_d]
            k = k.view(b, s_y, -1, self.head_dim)
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=input_pos)

            # View + expand + reshape bring num_kv_heads to num_heads for k and v
            # to match q.

            # k: [b, s_y, n_kv, 1, h_d]
            # v: [b, s_y, n_kv, 1, h_d]
            k = k.view(b, s_y, self.num_kv_heads, 1, self.head_dim)
            v = v.view(b, s_y, self.num_kv_heads, 1, self.head_dim)

            # If needed, expand the key and value tensors to have the same shape
            # as the query tensor by copying values across the relevant dim
            if self.num_heads != self.num_kv_heads:
                k = k.expand(b, s_y, self.num_kv_heads, q_per_kv, self.head_dim)
                v = v.expand(b, s_y, self.num_kv_heads, q_per_kv, self.head_dim)

            # [b, s, n_h, h_d]
            k = k.reshape(b, s_y, -1, self.head_dim)
            v = v.reshape(b, s_y, -1, self.head_dim)

            # [b, n_h, s, h_d]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Normalize k
            if self.k_norm is not None:
                k = self.k_norm(k)

            # Update key-value cache
            if self.kv_cache is not None and self.cache_enabled:
                k, v = self.kv_cache.update(k, v)

        output = self._attention_call(
            q,
            k,
            v,
            mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=self.kv_cache is None and mask is None and self.is_causal,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)
        return self.output_proj(output)


class TransformerSelfAttentionLayer(nn.Module):
    """
    Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (Optional[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        sa_scale (Optional[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        sa_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.sa_scale = sa_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): this parameter is ignored in this layer.
            decoder_max_seq_len (int): maximum cache sequence length.
        """
        self.attn.setup_cache(batch_size, dtype, max_seq_len=decoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup on ``self.attn``.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_setup`.
        """
        return self.attn.kv_cache is not None

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches on ``self.attn`` are enabled.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_enabled`.
        """
        return self.attn.cache_enabled

    def reset_cache(self):
        """Reset the key value caches."""
        self.attn.reset_cache()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            **kwargs (Dict): transformer layer inputs not relevant to self attention.

        Returns
        -------
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h = self.sa_norm(x)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.sa_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out


class TransformerCrossAttentionLayer(nn.Module):
    """
    Cross attention Transformer layer following the same conventions as the TransformerSelfAttentionLayer.
    Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        ca_norm (Optional[nn.Module]): Normalization to be applied before cross-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        ca_scale (Optional[nn.Module]): Module to scale cross-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.

    Raises
    ------
        AssertionError: if attn.pos_embeddings is set.
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        ca_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        ca_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if attn.pos_embeddings is not None:
            raise AssertionError(
                "Doesn't support positional embeddings for cross attention, \
                because q and k are different sequences."
            )
        self.attn = attn
        self.mlp = mlp
        self.ca_norm = ca_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.ca_scale = ca_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): maximum cache sequence length.
            decoder_max_seq_len (int): this parameter is ignored in this layer.
        """
        self.attn.setup_cache(batch_size, dtype, encoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup on ``self.attn``.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_setup`.
        """
        return self.attn.kv_cache is not None

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches on ``self.attn`` are enabled.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_enabled`.
        """
        return self.attn.cache_enabled

    def reset_cache(self):
        """Reset the key value caches."""
        self.attn.reset_cache()

    def _skip_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Some tokens in x may not attend to any encoder inputs
        due to the cross attention mask (encoder_mask). This results in
        a full row of the attention matrix being masked out.

        In the example below, the word "the" is masked from every embedding.
        The False value means a token can't attend to an embedding.

        .. code-block:: text

            |emb||emb||emb|
        |The| F    F    F
        |red| T    F    T
        |car| F    T    T

        This results in no inputs into the softmax layer which causes a NaN.
        The skip mask is used to mask the outputs of attention and
        mlp resulting in the token being skipped.

        The above example would result in a skip mask of: [[True], [False], [False]]
        which specifies which tokens to fully mask out.

        """
        # no skip_mask if no masking
        if mask is None:
            return None
        # negate mask and convert to boolean mask
        if mask.dtype == torch.bool:
            mask = ~mask
        else:
            mask = torch.isneginf(mask)
        # True where all elements in a row are True
        mask = torch.all(mask, dim=-1, keepdim=True)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape
                [batch_size x token_sequence x embed_dim]
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position i,j means token i can attend
                to embedding j in the decoder. Mask has shape [batch_size x token_sequence x embed_sequence].
                Default is None.
            **kwargs (Dict): transformer layer inputs not relevant to self attention.

        Returns
        -------
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # During decoding, it's possible encoder_input is None because the embeds
        # are already stored in the kv cache.
        empty_cache = not self.caches_are_enabled() or self.attn.kv_cache.size == 0
        # Skip cross attention when no secondary input as it's primary purpose
        # is to attend between x and encoder_input.
        if encoder_input is None and empty_cache:
            return x

        # A mask of tokens (x) with no encoder_input
        skip_mask = self._skip_mask(encoder_mask)
        if encoder_mask is not None:
            # TODO: remove after PyTorch 2.5 is released
            # This unmasks the skipped rows to avoid NaNs in SDPA Softmax backward
            # This doesn't affect the output since outputs are masked out later
            encoder_mask = encoder_mask.masked_fill(skip_mask, True)

        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        # TODO: Add support for sample packing and bring back input_pos
        attn_out = self.attn(self.ca_norm(x), encoder_input, mask=encoder_mask)
        if skip_mask is not None:
            attn_out = attn_out.masked_fill(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.ca_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))
        if skip_mask is not None:
            mlp_out = mlp_out.masked_fill(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out


## transformer


class Llama3ScaledRoPE(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864 with additional
    scaling from https://github.com/meta-llama/llama-models/blob/dc42f22a3b05502e7296402b019a51f57fa045c9/models/llama3_1.

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Default scaling factors are from the following Meta-Llama code:
    https://github.com/meta-llama/llama-models/blob/dc42f22a3b05502e7296402b019a51f57fa045c9/models/llama3_1/api/model.py#L41

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
        scale_factor (int): scaling factor for theta. Default: 8
        low_freq_factor (int): low frequency factor for scaling theta. Default: 1
        high_freq_factor (int): high frequency factor for scaling theta. Default: 4
        old_context_len (int): old context length for scaling theta. Default: 8192
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        scale_factor: int = 8,
        low_freq_factor: int = 1,
        high_freq_factor: int = 4,
        old_context_len: int = 8192,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.is_cache_built = False
        self.rope_init()

    def rope_init(self):
        """
        Warning: this is called in recipes before torch.compile,
        so that the cache is built in advance.
        """
        freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))

        # If we're on meta device return early.
        # We can't apply scaling until freqs is filled with real data
        if freqs.is_meta:
            return

        theta = self.apply_scaling(
            freqs,
            self.scale_factor,
            self.low_freq_factor,
            self.high_freq_factor,
            self.old_context_len,
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)
        self.is_cache_built = True

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def apply_scaling(
        self,
        freqs: torch.Tensor,
        scale_factor: int,
        low_freq_factor: int,
        high_freq_factor: int,
        old_context_len: int,
    ):
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns
        -------
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        Raises
        ------
            RuntimeError: if RoPE cache is not initialized prior to forward call
        """
        if not self.is_cache_built:
            raise RuntimeError("RoPE cache is not built. Please call rope_init() first.")

        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Return a list of ``n`` identical layers.

    Args:
        module (nn.Module): module to be cloned
        n (int): number of clones

    Returns
    -------
        nn.ModuleList: list of ``n`` identical layers
    """
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder derived from the Llama2 architecture.

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layers (Union[nn.Module, List[nn.Module], nn.ModuleList]): A single transformer Decoder layer, an
            nn.ModuleList of layers or a list of layers. It is recommended to use an nn.ModuleList.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            :func:`~torchtune.modules.KVCache`
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the :func:`~torchtune.modules.KVCache`
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        output (Union[nn.Linear, Callable]): Callable that applies a linear transformation to the output of
            the decoder.
        num_layers (Optional[int]): Number of Transformer Decoder layers, only define when
            layers is not a list.
        output_hidden_states (Optional[List[int]]): List of layers (indices) to include in the output

    Raises
    ------
        AssertionError: num_layers is set and layer is a list
        AssertionError: num_layers is not set and layer is an nn.Module

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        *,
        tok_embeddings: nn.Embedding,
        layers: Union[nn.Module, List[nn.Module], nn.ModuleList],
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: Union[nn.Linear, Callable],
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if isinstance(layers, nn.ModuleList):
            pass
        elif isinstance(layers, list):
            layers = nn.ModuleList(layers)
        else:
            if not isinstance(layers, nn.Module):
                raise AssertionError("num_layers is defined, layers must be a module")
            if num_layers is None:
                raise AssertionError("num_layers is not defined, layers must be a list")
            layers = _get_clones(layers, num_layers)

        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output = output
        self.output_hidden_states = output_hidden_states or []
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None
        self.num_output_chunks = 0

        # attributes for KV caches during inference
        self.encoder_max_cache_seq_len = None
        self.decoder_max_cache_seq_len = None

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe.
        """
        self.num_output_chunks = num_output_chunks

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ):
        """
        Sets up key-value attention caches for inference. For each layer in ``self.layers``:
            - :class:`~torchtune.modules.TransformerSelfAttentionLayer` will use ``decoder_max_seq_len``.
            - :class:`~torchtune.modules.TransformerCrossAttentionLayer` will use ``encoder_max_seq_len``.
            - :class:`~torchtune.modules.model_fusion.FusionLayer` will use ``decoder_max_seq_len`` and ``encoder_max_seq_len``.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (Optional[int]): maximum encoder cache sequence length.
            decoder_max_seq_len (Optional[int]): maximum decoder cache sequence length.
        """
        has_encoder_layers = any(isinstance(m, TransformerCrossAttentionLayer) for m in self.modules())
        has_decoder_layers = any(isinstance(la, TransformerSelfAttentionLayer) for la in self.layers)

        if has_encoder_layers:
            if encoder_max_seq_len is not None:
                self.encoder_max_cache_seq_len = encoder_max_seq_len
            else:
                self.encoder_max_cache_seq_len = self.max_seq_len

        if has_decoder_layers:
            if decoder_max_seq_len is not None:
                self.decoder_max_cache_seq_len = decoder_max_seq_len
            else:
                self.decoder_max_cache_seq_len = self.max_seq_len

        for layer in self.layers:
            layer.setup_caches(
                batch_size,
                dtype,
                encoder_max_seq_len=self.encoder_max_cache_seq_len,
                decoder_max_seq_len=self.decoder_max_cache_seq_len,
            )

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.layers[0].caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.layers[0].caches_are_enabled()

    def reset_caches(self):
        """
        Resets KV-cache buffers on relevant attention modules to zero, and reset cache positions to zero,
        without deleting or reallocating cache tensors.

        Raises
        ------
            RuntimeError: if KV-caches are not setup. Use :func:`~torchtune.modules.TransformerDecoder.setup_caches` to
                setup caches first.
        """
        if not self.caches_are_enabled():
            raise RuntimeError("Key value caches are not setup. Call model.setup_caches first.")

        for layer in self.layers:
            layer.reset_cache()

    @torch.compiler.disable
    def chunked_output(self, last_hidden_state: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply output projection in chunks. This should be applied in conjunction with
        :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss` as upcasting to fp32 is done there.

        To use this method, you should first call
        :func:`~torchtune.modules.TransformerDecoder.set_num_output_chunks`.

        Args:
            last_hidden_state (torch.Tensor): last hidden state of the decoder, having shape
                [b, seq_len, embed_dim].

        Returns
        -------
            List[torch.Tensor]: List of num_chunks output tensors, each with shape
                [b, seq_len/num_chunks, out_dim], where out_dim is usually the vocab size.
        """
        return [self.output(chunk) for chunk in last_hidden_state.chunk(self.num_output_chunks, dim=1)]

    def _validate_inputs(
        self,
        seq_len: int,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        """
        Validates inputs for ``forward``.
        Args:
            seq_len (int): Input tensor sequence length.
            mask (Optional[torch.Tensor]): Attention mask used for inference and for sequence packing.
            encoder_input (Optional[torch.Tensor]): Encoder input for cross-attention.
            encoder_mask (Optional[torch.Tensor]): Encoder attention mask for cross-embedding attention.
            input_pos (Optional[torch.Tensor]): Input tensor position IDs.

        Raises
        ------
            ValueError: if seq_len of x is bigger than max_seq_len
            ValueError: if the model has caches which have been setup with self-attention layers and ``mask`` is not provided.
            ValueError: if the model has caches which have been setup with encoder layers and ``encoder_mask`` is not provided.
            ValueError: if the model has caches which have been setup ``input_pos`` is not provided.
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller " f"than max_seq_len ({self.max_seq_len})"
            )

        if self.caches_are_enabled():
            if mask is None:
                raise ValueError(
                    "KV-caches for self-attention layers are setup for inference mode, causal masks must be provided!"
                    " Use the `mask` arg to provide a causal mask."
                )

            if encoder_input is not None and encoder_mask is None:
                raise ValueError(
                    "KV-caches for cross-attention/fusion layers are setup for inference mode and you seem to be using"
                    " encoder_input, causal masks must be provided! Use the `encoder_mask` arg to provide a causal mask."
                )

            if input_pos is None:
                raise ValueError("KV-caches are setup for inference mode, input positions must be provided!")

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. This parameter is required during inference if caches have been setup.
                Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position ``i,j`` means token ``i`` can attend
                to embedding ``j`` in the decoder. Mask has shape ``[b x s x s_e]``. Default is None,
                but this is required during inference if the model has been setup with any layers
                which use encoder embeddings and caches have been setup.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                This parameter is required during inference if caches have been setup. Default is None.

        Returns
        -------
            Union[torch.Tensor, List[torch.Tensor]]: output tensor with shape ``[b x s x v]`` or a list of layer
                output tensors defined by ``output_hidden_states`` with the
                final output tensor appended to the list.

        Note:
            At the very first step of inference, when the model is provided with a prompt,
            ``input_pos`` should contain the positions of all of the tokens in the prompt.
            For a single-batch prompt, or a batch of prompts with identical lengths, this
            will be ``torch.arange(prompt_length)``. For a batch of varying-length prompts,
            shorter prompts are left-padded and position ids are correspondingly right-shifted,
            thus positional ids should be of shape ``[b, padded_prompt_length]``.
            This is because we will need to retrieve the positional embeddings for each input id.
            In the subsequent steps, if the model has been setup with KV-caches, ``input_pos`` will contain
            the position(s) of the current token(s) ``torch.tensor([padded_prompt_length])``. Otherwise,
            ``input_pos`` will contain all the position ids up to the current token.

        Shape notation:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        seq_len = tokens.shape[1]

        self._validate_inputs(
            seq_len,
            mask=mask,
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )

        # shape: [b, s, d]
        h = self.norm(h)

        if self.num_output_chunks > 0:
            output = self.chunked_output(h)
        else:
            # shape: [b, seq_len, out_dim]
            output = self.output(h).float()

        # Output list if hidden states are requested, otherwise just the output
        # TODO: always output a list to have a consistent output type
        output = output if not hidden else [*hidden, output]
        return output


# -----------------------------------------------------------------------------

# transform


def _get_max_res_without_distortion(
    image_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Determines the maximum resolution to which an image can be resized to without distorting its
    aspect ratio, based on the target resolution.

    For example, if image_size = (200,400) and target_size = (600,800),
    scale_h = 600/200 = 3
    scale_w = 800/400 = 2
    So the maximum that we can upscale without distortion is min(scale_h, scale_w) = 2

    Since scale_w is the limiting side, then new_w = target_w, and new_h = old_h*scale_w

    Args:
        image_size (Tuple[int, int]): The original resolution of the image.
        target_size (Tuple[int, int]): The desired resolution to fit the image into.

    Returns
    -------
        Tuple[int, int]: The optimal dimensions to which the image should be resized.

    Examples
    --------
        >>> _get_max_res_without_distortion([200, 300], target_size = (450, 200))
        (133, 200)
        >>> _get_max_res_without_distortion([800, 600], target_size = (450, 1300))
        (450, 337)
    """
    original_height, original_width = image_size
    target_height, target_width = target_size

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.floor(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.floor(original_width * scale_h), target_width)

    return new_height, new_width


def _pad_image_top_left(
    image: torch.Tensor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Places the image at the top left of the canvas and pads with 0 the right and bottom
    to fit to the target resolution. If target_size < image_size, it will crop the image.

    Args:
        image (torch.Tensor): The input image tensor in the format [..., H, W].
        target_size (Tuple[int, int]): The desired resolution to fit the image into in the format [height, width].

    Returns
    -------
        torch.Tensor: The padded image tensor in the format [..., H, W].
    """
    image_size = image.shape[-2:]

    height, width = image_size
    target_height, target_width = target_size

    pad_x = target_width - width
    pad_y = target_height - height

    padding = [0, 0, pad_x, pad_y]
    return F.pad(inpt=image, padding=padding)


def resize_with_pad(
    image: torch.Tensor,
    target_size: Tuple[int, int],
    resample: torchvision.transforms.InterpolationMode,
    max_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Resizes and pads an image to target_size without causing distortion.
    The user can set max_size to limit upscaling when target_size exceeds image_size.

    Args:
        image (torch.Tensor): The input image tensor in the format [..., H, W].
        target_size (Tuple[int, int]): The desired resolution to fit the image into in the format [height, width].
        resample (torchvision.transforms.InterpolationMode): Resampling method used when resizing images.
            Supports torchvision.transforms.InterpolationMode.NEAREST, InterpolationMode.NEAREST_EXACT,
            InterpolationMode.BILINEAR and InterpolationMode.BICUBIC.
        max_size (Optional[int]): The maximum size to upscale the image to.
            If None, will upscale up to target_size.

    Returns
    -------
        torch.Tensor: The resized and padded image tensor in the format [..., H, W].

    Examples
    --------
        Example 1: The image will be upscaled from (300, 800) to (448, 1194), since 448 is the limiting side,
        and then padded from (448, 1194) to (448, 1344).

            >>> max_size = None
            >>> image = torch.rand([3, 300, 800])
            >>> target_size = (448, 1344)
            >>> resample = torchvision.transforms.InterpolationMode.BILINEAR
            >>> output = resize_with_pad(image, target_size, resample, max_size)

        Example 2: The image will stay as is, since 800 > 600, and then padded from (300, 800) to (448, 1344).

            >>> max_size = 600
            >>> image = torch.rand([3, 300, 800])
            >>> target_size = (448, 1344)
            >>> resample = torchvision.transforms.InterpolationMode.BILINEAR
            >>> output = resize_with_pad(image, target_size, resample, max_size)

        Example 3: The image will be downscaled from (500, 1000) to (224, 448),
        and padded from (224, 448) to (448, 448).

            >>> max_size = 600
            >>> image = torch.rand([3, 500, 1000])
            >>> target_size = (448, 488)
            >>> resample = torchvision.transforms.InterpolationMode.BILINEAR
            >>> output = resize_with_pad(image, target_size, resample, max_size)

    """
    image_height, image_width = image.shape[-2:]
    image_size = (image_height, image_width)

    # If target_size requires upscaling, we might want to limit the upscaling to max_size
    if max_size is not None:
        new_target_height = min(max(image_height, max_size), target_size[0])
        new_target_width = min(max(image_width, max_size), target_size[1])
        target_size_resize = (new_target_height, new_target_width)
    else:
        target_size_resize = target_size

    # resize to target_size while preserving aspect ratio
    new_size_preserving_aspect_ratio = _get_max_res_without_distortion(
        image_size=image_size,
        target_size=target_size_resize,
    )

    image = F.resize(
        inpt=image,
        size=list(new_size_preserving_aspect_ratio),
        interpolation=resample,
        antialias=True,
    )

    image = _pad_image_top_left(image=image, target_size=target_size)

    return image


def tile_crop(image: torch.Tensor, tile_size: int) -> torch.Tensor:
    """
    Divides a tensor into equally sized tiles. The tensor should be divisible by tile_size.

    Args:
        image (torch.Tensor): Input image to crop into tiles.
        tile_size (int): Size of each tile.

    Returns
    -------
        torch.Tensor: torch.Tensor of shape [num_tiles, channel_size, tile_size, tile_size]

    Examples
    --------
        >>> image = torch.rand(3, 200, 300)
        >>> tiles = tile_crop(image, tile_size=50)
        >>> tiles.shape # 4x6 = 24 tiles
        torch.Size([24, 3, 50, 50])

        >>> image = torch.rand(3, 400, 600)
        >>> tiles = tile_crop(image, tile_size=200)
        >>> tiles.shape # 2x3 = 6 tiles
        torch.Size([6, 3, 200, 200])
    """
    channel_size, height, width = image.shape

    # assert sizes are divisible
    assert (
        height % tile_size == 0 and width % tile_size == 0
    ), f"Image size {height}x{width} is not divisible by tile size {tile_size}"

    # Reshape to split height and width into tile_size blocks
    tiles_height = height // tile_size
    tiles_width = width // tile_size

    reshaped = image.view(channel_size, tiles_height, tile_size, tiles_width, tile_size)

    # Transpose to bring tiles together
    # We want [tiles_height, tiles_width, channel_size, tile_size, tile_size]
    transposed = reshaped.permute(1, 3, 0, 2, 4)

    # Flatten the tiles
    tiles = transposed.contiguous().view(tiles_height * tiles_width, channel_size, tile_size, tile_size)

    return tiles


def get_canvas_best_fit(
    image: torch.Tensor, possible_resolutions: torch.Tensor, resize_to_max_canvas: bool
) -> Tuple[int, int]:
    """
    Determines the best canvas possible from a list of possible resolutions to
    resize an image to, without distortion.

    For each possible resolution, calculates the scaling factors for
    width and height, and selects the smallest one, which is the limiting side.
    E.g. if to match a canvas shape you have to upscale an image's height by 2x, and width by 1.5x,
    then the maximum upscaling without distortion is min(2, 1.5) = 1.5.

    If there are multiple canvases that satisfy the conditions,
    we pick the one with the lowest area to minimize padding.

    Args:
        image (torch.Tensor): The image we want to fit into a canvas.
        possible_resolutions (torch.Tensor): A tensor of shape (N, 2) where each
            row represents a possible canvas.
        resize_to_max_canvas (bool): If True, pick the canvas that allows maximum scaling.
            If False, pick the canvas that minimizes downscaling, including no downscaling at all.

    Returns
    -------
        Tuple[int, int]: The best resolution to fit the image into.

    Examples
    --------
        >>> image = torch.rand(3, 200, 300)
        >>> possible_resolutions = torch.tensor([
        ...     [224, 672],
        ...     [672, 224],
        ...     [224, 448],
        ...     [448, 224],
        ...     [224, 224]
        ... ])
        >>> get_canvas_best_fit(image, possible_resolutions, resize_to_max_canvas=False)
        (224, 448)

        In the example above, we calculate the scaling factors for each possible resolution

        >>> scale_height = torch.tensor([1.1200, 3.3600, 1.1200, 2.2400, 1.1200])
        >>> scale_width = torch.tensor([2.2400, 0.7467, 1.4933, 0.7467, 0.7467])
        >>> scales = torch.tensor([1.1200, 0.7467, 1.1200, 0.7467, 0.7467])

        Two options have scaling_factor > 1, since resize_to_max_canvas is False, we pick the smallest

        >>> upscaling_options = torch.tensor([1.1200, 1.1200])
        >>> selected_scale = torch.tensor(1.1200)

        There are two possible options, so we pick the one with the smallest area

        >>> areas = torch.tensor([150528, 100352])  # for resolutions [672, 224] and [224, 448], respectively
        >>> optimal_canvas = torch.tensor([224, 448])  # resolution with the smallest area
    """
    original_height, original_width = image.shape[-2:]

    # possible resolutions heights/widths
    target_heights, target_widths = (
        possible_resolutions[:, 0],
        possible_resolutions[:, 1],
    )

    # scaling factors to resize the image without distortion
    scale_w = target_widths / original_width
    scale_h = target_heights / original_height

    # get limiting side scaling -> no distortion
    scales = torch.where(scale_w > scale_h, scale_h, scale_w)

    # filter only scales that allow upscaling
    upscaling_options = scales[scales >= 1]
    if len(upscaling_options) > 0:
        if resize_to_max_canvas:
            selected_scale = torch.max(upscaling_options)
        else:
            selected_scale = torch.min(upscaling_options)
    else:
        # no upscaling possible,
        # get the minimum downscaling (max scale for scales<1)
        downscaling_options = scales[scales < 1]
        selected_scale = torch.max(downscaling_options)

    # get all resolutions that support this scaling factor,
    # e.g. you can upscale to 224x224, 224x448, 224x672 without distortion
    chosen_canvas = possible_resolutions[scales == selected_scale]

    # if there are multiple resolutions,
    # get the one with minimum area to reduce padding
    if len(chosen_canvas) > 1:
        areas = chosen_canvas[:, 0] * chosen_canvas[:, 1]
        optimal_idx = torch.argmin(areas)
        optimal_canvas = chosen_canvas[optimal_idx]
    else:
        optimal_canvas = chosen_canvas[0]

    return tuple(optimal_canvas.tolist())


def find_supported_resolutions(max_num_tiles: int, tile_size: int) -> List[Tuple[int, int]]:
    """
    Computes all combinations of resolutions, multiple of tile_size,
    that contain up to max_num_tiles. Useful for when dividing an image into tiles.

    For example, if we want at most 2 tiles per image, then we can support the
    following resolutions: (1x1, 1x2, 2x1) * tile_size

    Args:
        max_num_tiles (int): Maximum number of tiles.
        tile_size (int): Size of the side of the tile.

    Returns
    -------
        List[Tuple[int, int]]: List of possible resolutions as tuples (height, width).

    Examples
    --------
        >>> max_num_tiles = 4
        >>> tile_size = 224
        >>> find_supported_resolutions(max_num_tiles, tile_size)
        [(224, 896), (448, 448), (224, 224), (896, 224), (224, 672), (672, 224), (224, 448), (448, 224)]
    """
    # create dictionary {aspect_ratio: [resolution1, ..., resolution n]}
    # example {0.25: [(1,4)], 1.0: [(2,2), (1,1)], 4.0: [(4,1)]}
    asp_dict = defaultdict(list)
    for _tile_size in range(max_num_tiles, 0, -1):
        factors = sorted(_get_factors(_tile_size))
        asp_ratios = [(factor, _tile_size // factor) for factor in factors]
        for height, width in asp_ratios:
            ratio_float = height / width
            asp_dict[ratio_float].append((height, width))

    # get the resolutions multiplied by the tile_size
    possible_resolutions = []
    for _, resolution in asp_dict.items():
        for height, width in resolution:
            possible_resolutions.append((height * tile_size, width * tile_size))

    return possible_resolutions


def _get_factors(n: int) -> Set[int]:
    """
    Calculate all factors of a given number, i.e. a divisor that leaves no remainder.

    Args:
        n (int): The number to find factors for.

    Returns
    -------
        set: A set containing all factors of the number.

    Examples
    --------
        >>> _get_factors(n=12)
        {1, 2, 3, 4, 6, 12}
    """
    factors_set = set()

    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors_set.add(i)
            factors_set.add(n // i)
    return factors_set


class CLIPImageTransform:
    """
    This class accepts images of any size and dynamically resizes, pads, normalizes and tiles it
    based on the image aspect ratio and the number of image tiles we allow.

    The algorithm will NOT distort the image to fit a certain aspect ratio, because
    that leads to a significant degradation in image quality.

    The user can choose if they want to allow upscaling by using the flag ``resize_to_max_canvas``.

    For example, if an input image is of size 300x800, and we want to allow
    a maximum of 16 image tiles, with side 224px, then:

    If ``resize_to_max_canvas=False``, then:
    best_resolution = (448, 896) -> smallest canvas, up to 16 tiles, that doesn't require downscaling
    image is NOT resized
    image is padded (300, 800) -> 448,896
    Image is tiled 2x4, for a final output shape of (8, 3, 224, 224)

    If ``resize_to_max_canvas=True``, then:
    best_resolution = (448, 1344) # canvas that allows maximum upscaling, with minimum padding, up to 16 tiles
    image is resized without distortion (300,800) -> (448, 1194) #448 is the limiting side for the resize
    image is padded (448, 1194) -> (448, 1344)
    Image is tiled 2x5, for a final output shape of (10, 3, 224, 224)

    Args:
        image_mean (Optional[List[float]]): Mean values of each channel, used for normalization.
            Should be the same used for the pre-trained model. If None, no normalization is performed. Default None.
        image_std (Optional[List[float]]): Standard deviation values of each channel, used for normalization.
            Should be the same used for the pre-trained model. If None, no normalization is performed. Default None.
        possible_resolutions (Optional[List[Tuple[int, int]]]): List of possible resolutions as tuples (height, width).
            where each tuple represents a possible canvas to fit the image into when calling ``get_canvas_best_fit``.
            If None, this will be calculated using max_num_tiles and tile_size. Default None.
        tile_size (int): Size of the tiles to divide the image into. Default 224.
        max_num_tiles (Optional[int]): Only used if possible_resolutions is NOT given.
            Maximum number of tiles to break an image into.
            This will be used to generate possible_resolutions,
            e.g. [(224, 224), (224, 448), (448, 224)] if max_num_tiles = 2 and tile_size = 224.
            Default 4.
        dtype (torch.dtype): Data type of the output image. Default torch.bfloat16.
        resample (str): Resampling method used when resizing images. Supports any enum of
            ``torchvision.transforms.InterpolationMode``, e.g. "nearest", "nearest_exact", "bilinear", "bicubic".
            Default 'bilinear'.
        resize_to_max_canvas (bool): "If True, the image will be upscaled without distortion to fit the largest possible
            resolution from possible_resolutions.
            If False, it will pick the resolution that minimizes downscaling, including no downscaling at all.
            In this case, the image will only be upscaled if it's size < tile_size. Default False.

    Examples
    --------
        >>> image_transform = CLIPImageTransform(
        ...    image_mean=None,
        ...    image_std=None,
        ...    tile_size=224,
        ...    possible_resolutions=None,
        ...    max_num_tiles=4,
        ...    resample="bilinear",
        ...    resize_to_max_canvas=True,
        ...)
        >>> # create random image
        >>> image = (np.random.rand(100,200,3) * 255).astype(np.uint8)
        >>> image = PIL.Image.fromarray(image)
        >>> output = image_transform(image)
        >>> output['image'].shape # [num_tiles, num_channels, tile_size, tile_size]
        torch.Size([2, 3, 224, 224])
        >>> output['ar'] # image best fits the canvas 224x448
        torch.tensor([1,2])
    """

    def __init__(
        self,
        *,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        possible_resolutions: Optional[List[Tuple[int, int]]] = None,
        tile_size: int = 224,
        max_num_tiles: Optional[int] = 4,
        dtype: torch.dtype = torch.bfloat16,
        resample: str = "bilinear",
        resize_to_max_canvas: bool = False,
    ) -> None:
        # get_canvas_best_fit
        assert (
            possible_resolutions is not None or max_num_tiles is not None
        ), f"Either possible_resolutions or max_num_tiles must be given. Got {possible_resolutions=} and {max_num_tiles=}"

        # If possible_resolutions are not given, then calculate possible ones based on max_num_tiles
        if not possible_resolutions and max_num_tiles:
            possible_resolutions = find_supported_resolutions(max_num_tiles=max_num_tiles, tile_size=tile_size)
        else:
            possible_resolutions = possible_resolutions

        self.possible_resolutions = torch.tensor(possible_resolutions).reshape(-1, 2)
        logger.debug(
            f"Found possible_resolutions: {self.possible_resolutions}. Will fit the images into the canvas with best fit."
        )

        self.resize_to_max_canvas = resize_to_max_canvas

        # normalize
        assert (image_mean is None) == (
            image_std is None
        ), f"Need to provide both or none of image_mean and image_std. Got {image_mean=} and {image_std=}"
        self.mean = image_mean
        self.std = image_std

        # resize_with_pad
        self.max_size = None if resize_to_max_canvas else tile_size
        self.dtype = dtype
        self.resample = torchvision.transforms.InterpolationMode[resample.upper()]

        # tile_crop
        self.tile_size = tile_size
        self.tile_crop = tile_crop

    def __call__(self, sample: Mapping[str, Any], inference: bool = False) -> Mapping[str, Any]:
        """
        Apply image decoding and transformations to the "image" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with an "image" field containing
                a List[Message] to tokenize
            inference (bool): Whether the template is being used for inference or not.

        Returns
        -------
            Mapping[str, Any]: The sample with an updated "image" filed and added
                "aspect_ratio" field.
        """
        image = sample["image"]
        assert isinstance(image, Image.Image), "Input image must be a PIL image."

        # Make image torch.tensor((3, H, W), dtype=dtype), 0<=values<=1
        if hasattr(image, "mode") and image.mode == "RGBA":
            image = image.convert("RGB")
        image = F.to_image(image)
        image = F.grayscale_to_rgb_image(image)
        image = F.to_dtype(image, dtype=self.dtype, scale=True)

        # Find the best canvas to fit the image without distortion
        best_resolution = get_canvas_best_fit(
            image=image,
            possible_resolutions=self.possible_resolutions,
            resize_to_max_canvas=self.resize_to_max_canvas,
        )

        # resize without distortion + pad to fit best_resolution
        image = resize_with_pad(
            image=image,
            target_size=best_resolution,
            resample=self.resample,
            max_size=self.max_size,
        )

        # Normalize
        if self.mean:
            image = F.normalize(image, mean=self.mean, std=self.std)

        # Divide the image into equally sized tiles
        image = self.tile_crop(image=image, tile_size=self.tile_size)

        aspect_ratio = torch.tensor(best_resolution).reshape(-1) // self.tile_size

        sample.update(
            {
                "image": image,
                "aspect_ratio": aspect_ratio,
            }
        )

        return sample


class VisionCrossAttentionMask(Transform):
    """
    Computes the cross-attention mask for text + image inputs. Text tokens that
    participate in cross-attention with an image token will show True in the mask
    and follow the interleaved structure laid out in Fig. 7 of the Flamingo paper
    (https://arxiv.org/pdf/2204.14198):

        (1) Text tokens immediately following the image token up until the next image token
        (2) Consecutive image tokens attend to subsequent text tokens

    ::

             ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
        img1 │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │   │ │   │ │   │ │   │ │   │
             └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
             ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
        img2 │   │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │   │ │   │ │   │ │   │ │   │
             └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
             ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
        img3 │   │ │   │ │   │ │   │ │   │ │   │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │
             └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
            <img1> <img2>These  are   two  dogs. <img3> This   is    a    cat.


    Resultant mask is constructed per image and is of shape (text_seq_len, image_seq_len),
    where True indicates that the token outputted from the image encoder attends
    to the token in the text sequence in cross-attention. A list of these masks
    are returned with length equal to number of images in the sample.

    Args:
        tile_size (int): The size of the image tiles from the image transform
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for patch_size = 40, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        image_token_id (int): Token ID of the image special token.
        max_num_tiles (Optional[int]): Maximum number of tiles in an image, used to
            pad mask during inference. Defaults to None
    """

    def __init__(
        self,
        tile_size: int,
        patch_size: int,
        image_token_id: int,
        max_num_tiles: Optional[int] = None,
    ):
        patch_grid_size = tile_size // patch_size
        self.patches_per_tile = patch_grid_size**2
        self.image_token_id = image_token_id
        self.max_num_tiles = max_num_tiles

    def _get_image_attention_intervals(self, tokens: List[int]) -> List[List[int]]:
        """
        Returns a list of lists of the form [start, end) where start is the index
        of the current image token and end is the index of the next image token, exclusive.

        Args:
            tokens (List[int]): List of token IDs in the text sequence

        Returns
        -------
            List[List[int]]: List of lists of the form [start, end) indicating
                range of positions in text sequence that should attend to the image

        Example:
            >>> text = "<img1><img2>These are two dogs. <img3>This is a cat."
            >>> image_token_id = 1
            >>> tokens = [1, 1, 9673, 527, 1403, 12875, 13, 1, 1115, 374, 264, 8415]
            >>> transform = VisionCrossAttentionMask(tile_size=400, patch_size=40, image_token_id=1)
            >>> intervals = transform._get_image_attention_intervals(tokens)
            >>> print(intervals)
            [[0, 7], [1, 7], [7, 12]]
        """
        end = len(tokens)
        vision_token_locations = [i for i, token in enumerate(tokens) if token == self.image_token_id]
        # Return empty list if there are no images
        if len(vision_token_locations) == 0:
            return []
        # If there is only one image, it will attend to subsequent text until end
        if len(vision_token_locations) == 1:
            return [[vision_token_locations[0], end]]

        # Construct intervals from previous image token to next image token
        vision_masks = [
            [tok_idx_prev, tok_idx_next]
            # Offset by one to get consecutive indices
            for tok_idx_prev, tok_idx_next in zip(vision_token_locations[:-1], vision_token_locations[1:], strict=False)
        ]
        # Last image will attend to subsequent text until end
        vision_masks.append([vision_token_locations[-1], end])

        # If there are consecutive vision tokens, they should all attend to the
        # same subsequent text
        last_mask_end = vision_masks[-1][1]
        for vision_mask in vision_masks[::-1]:
            if vision_mask[0] == vision_mask[1] - 1:
                vision_mask[1] = last_mask_end
            last_mask_end = vision_mask[1]
        return vision_masks

    def __call__(self, sample: Mapping[str, Any], inference: bool = False) -> Mapping[str, Any]:
        """
        Generates the vision cross-attention mask for the given sample based on
        the image token locations interleaved in the text sequence.

        Args:
            sample (Mapping[str, Any]): Sample dict containing the following keys:
                - tokens (List[int]): List of token IDs in the text sequence. Number of
                    image token IDs in the sequence must match the number of images.
                - images (List[torch.Tensor]): List of image Tensors post-tiling of shape
                    (n_tiles, c, h, w) each.
            inference (bool): Whether the template is being used for inference or not.

        Returns
        -------
            Mapping[str, Any]: sample with a new key encoder_mask, with a mask per image with shape
                (text_seq_len, image_seq_len) where text_seq_len == len(tokens) and
                image_seq_len == max_tiles * (patches_per_tile + 1). These masks get padded and concatenated
                in the batch collator.

        Raises
        ------
            RuntimeError: if the number of images in the batch does not match the number of image tokens in the batch.
        """
        tokens, images = sample["tokens"], sample["encoder_input"]["images"]
        # One sample can have multiple images - verify the number of image tokens
        # is the same
        n_img = len(images)
        intervals = self._get_image_attention_intervals(tokens)
        if len(intervals) != n_img:
            raise RuntimeError(
                f"The number of image tokens ({len(intervals)}) does not match the number of images ({n_img})."
            )

        # Create mask for each individual image based on its number of tokens,
        # which can vary based on number of tiles since they are not yet tile padded.
        # The masks are padded and concatenated together in the batch collator
        text_seq_len = len(tokens)
        max_image_size = None
        if inference and self.max_num_tiles is not None:
            max_image_size = self.max_num_tiles * (self.patches_per_tile + 1)
        masks = []
        for image_num, interval in enumerate(intervals):
            # Identify what part of text sequence should be attended
            start, end = interval
            # Compute this image's number of tokens based on num tiles, patches per tile
            n_tiles = images[image_num].shape[0]
            image_seq_len = n_tiles * (self.patches_per_tile + 1)  # +1 for CLS token
            # Mask will be block of 1s at the corresponding interval in the text.
            # It is not a causal block because all the image tokens correspond
            # to a single image, so text tokens attend to all the image's tokens.
            # The mask is text_seq_len x mask_image_size if defined, otherwise
            # it uses current text/image sequence lengths.
            mask = torch.zeros(text_seq_len, max_image_size or image_seq_len, dtype=torch.bool)
            mask[start:end, :image_seq_len] = True
            masks.append(mask)

        sample.update({"encoder_mask": masks})
        return sample


class Llama3VisionTransform(ModelTokenizer, Transform):
    """
    This transform combines the transforms for the different modalities of Llama 3.2 Vision. It
    is made up of the following transforms:
    - :class:`torchtune.models.llama3.Llama3Tokenizer`
    - :class:`torchtune.models.clip.CLIPImageTransform`
    - :class:`torchtune.modules.transforms.VisionCrossAttentionMask`

    This transform can be used as a drop-in replacement for tokenizers in recipes and generation
    but handles additional transformations from the `__call__` method.

    Args:
        path (str): Path to pretrained tiktoken tokenizer file.
        tile_size (int): Size of the tiles to divide the image into.
        patch_size (int): Size of the patches used in the CLIP vision tranformer model. This is
            used to calculate the number of image embeddings per image.
        max_num_tiles (int): Only used if possible_resolutions is NOT given.
            Maximum number of tiles to break an image into.
            This will be used to generate possible_resolutions,
            e.g. [(224, 224), (224, 448), (448, 224)] if max_num_tiles = 2 and tile_size = 224.
            Default 4.
        special_tokens (Optional[Dict[str, int]]): mapping containing special text tokens and
            their registered token IDs. If left as None, this will be set to the canonical
            Llama3 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        image_mean (Optional[Tuple[float, float, float]]): Mean values of each channel, used for normalization.
        image_std (Optional[Tuple[float, float, float]]): Standard deviations for each channel, used for normalization.
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens. Default is None.

    Examples
    --------
        >>> model_transform = Llama3VisionTransform("/path/to/tokenizer.model", tile_size=224, patch_size=14)
        >>> transformed_data = model_transform({"messages": user_message, "images": [img1, img2]})
        >>> print(transformed_data["tokens"])
        [1, 31587, 29644, 102, 2]
        >>> print(transformed_data["images"][0].shape)
        torch.Size([4, 3, 224, 224])
    """

    def __init__(
        self,
        path: str,
        *,
        tile_size: int,
        patch_size: int,
        max_num_tiles: int = 4,
        special_tokens: Optional[Dict[str, int]] = None,
        max_seq_len: Optional[int] = None,
        image_mean: Optional[Tuple[float, float, float]] = None,
        image_std: Optional[Tuple[float, float, float]] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        self.tokenizer = llama3_tokenizer(
            path,
            special_tokens_path=special_tokens,
            max_seq_len=max_seq_len,
            prompt_template=prompt_template,
        )
        self.transform_image = CLIPImageTransform(
            image_mean=image_mean,
            image_std=image_std,
            tile_size=tile_size,
            possible_resolutions=None,
            max_num_tiles=max_num_tiles,
            resample="bilinear",
            resize_to_max_canvas=False,
        )
        self.xattn_mask = VisionCrossAttentionMask(
            tile_size=tile_size,
            patch_size=patch_size,
            image_token_id=self.tokenizer.image_id,
            max_num_tiles=max_num_tiles,
        )

        self.stop_tokens = self.tokenizer.stop_tokens
        self.max_seq_len = max_seq_len
        self.image_seq_len = max_num_tiles * (self.xattn_mask.patches_per_tile + 1)
        self.prompt_template = prompt_template
        self.pad_id = self.tokenizer.pad_id

    @property
    def base_vocab_size(self) -> int:
        return self.tokenizer.base_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        return self.tokenizer.encode(text=text, add_bos=add_bos, add_eos=add_eos)

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            truncate_at_eos (bool): Whether to truncate the string at the end of
                sequence token. Default is True.
            skip_special_tokens (bool): Whether to show or skip special tokens in the decoded string.
                Default is True.

        Returns
        -------
            str: The decoded string.
        """
        return self.tokenizer.decode(
            token_ids,
            truncate_at_eos=truncate_at_eos,
            skip_special_tokens=skip_special_tokens,
        )

    def tokenize_message(
        self,
        message: Message,
        tokenize_header: bool = True,
        tokenize_end: bool = True,
    ) -> List[int]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            tokenize_header (bool): Whether to prepend a tokenized header to the message.
            tokenize_end (bool): Whether to append eot or eom id at the end of the message.

        Returns
        -------
            List[int]: The list of token ids.
        """
        return self.tokenizer.tokenize_message(
            message=message,
            tokenize_header=tokenize_header,
            tokenize_end=tokenize_end,
        )

    def tokenize_messages(
        self,
        messages: List[Message],
        add_eos: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            add_eos (bool): Wether to add the tokenizer's eos_id. Default True.

        Returns
        -------
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        return self.tokenizer.tokenize_messages(
            messages=messages,
            add_eos=add_eos,
        )

    def __call__(self, sample: Mapping[str, Any], inference: bool = False) -> Mapping[str, Any]:
        """
        Apply image decoding, transformations and tokenization to messages in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field.
            inference (bool): Whether to run in inference mode. Default is True.

        Returns
        -------
            Mapping[str, Any]: The transformed sample with the following fields:
                - tokens: List[int] of tokenized messages
                - mask: List[bool] of masks for the tokenized messages
                - encoder_input: Dict[str, Any] of transformed images
                - encoder_mask: List[bool] of masks for the transformed images
        """
        encoder_input = {"images": [], "aspect_ratio": []}
        messages = sample["messages"]
        for message in messages:
            for image in message.get_media():
                out = self.transform_image({"image": image}, inference=inference)
                encoder_input["images"].append(out["image"])
                encoder_input["aspect_ratio"].append(out["aspect_ratio"])

        sample["encoder_input"] = encoder_input
        sample = self.tokenizer(sample, inference=inference)
        sample = self.xattn_mask(sample, inference=inference)
        return sample


# -----------------------------------------------------------------------------

# model


def register_fusion_module(module: nn.Module):
    """Add the method fusion_params to an nn.Module that
    marks all of the Modules parameters as fusion params.
    This can be used for a layer or an entire model that is
    added to combine two or more pretrained models.

    For example, you might want to add a projection head
    head onto an encoder to learn a projection from the
    pre-trained encodings to the decoder's embedding space. This
    is typical with both Deep Fusion and Early Fusion models.

    Example:
        >>> projection_head = FeedForward(...)
        >>> register_fusion_module(projection_head))
        >>> encoder = nn.Sequential(clip_vit_224(), projection_head)

    Args:
        module (nn.Module): module to add the fusion_params method to
    """

    def fusion_params(self) -> List[str]:
        """
        Return parameters of fusion layer.
        """
        return [k for k, v in self.named_parameters()]

    module.fusion_params = functools.partial(fusion_params, module)


def get_fusion_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    """
    Return the subset of parameters from a model that correspond to fused
    modules. Assumes that any fusion class has defined the
    :func:`~torchtune.modules.model_fusion.FusionLayer.fusion_params` method.

    Args:
        model (nn.Module): Instance of model class containing some
            fusion params.

    Returns
    -------
        Dict[str, nn.Parameter]: the subset of model's state dict containing
            only adapter parameters.

    """
    fusion_params = {}
    for k, v in model.named_modules():
        if hasattr(v, "fusion_params") and callable(v.fusion_params):
            current_fusion_params = v.fusion_params()
            for n, p in v.named_parameters(recurse=True):
                if n in current_fusion_params:
                    full_key = f"{k}.{n}" if k else n
                    fusion_params.update({full_key: p})
                    current_fusion_params.remove(n)
            assert current_fusion_params == [], f"Fusion params {current_fusion_params} not converted"
    return fusion_params


class FusionLayer(nn.Module):
    """Fusion layer as introduced in `Flamingo: a Visual Language Model for Few-Shot Learning <https://arxiv.org/abs/2204.14198>`_.

    Deep Fusion model architectures combine pretrained encoder models with pretrained
    language models by infusing the encoder outputs into the middle layers of the LLM.
    This allows the language model to interpret the enocder outputs as text and
    "understand" any modality for which you can train an encoder. To enable the language model
    to adapt to the encoder outputs, the FusionLayer fuses a new learnable layer to an existing
    decoder (language model) layer. This additional layer can take the encoder embeddings and
    learn to combine them with the token embeddings from the decoder. The module supports fusing
    the new layer before or after the original, in Flamingo the new layer is fused before the original.

    The original layer is wrapped in FusionLayer such that it maintains its original state_dict
    key and the pre-trained checkpoint isn't broken. The new layer parameters are available
    through ``fusion_params`` to separately control if they're trainable or not.

    Example:
        >>> # Original decoder style transformer
        >>> layer = nn.TransformerSelfAttentionLayer(...)
        >>> model = TransformerDecoder(layers=layer, num_layers=32, ...)
        >>>
        >>> # Fuse a cross attention layer to each self attention layer to adapt for the encoder
        >>> fusion_layer = nn.TransformerCrossAttentionLayer(...)
        >>> fused_layer = FusionLayer(layer, fusion_layer)
        >>> model = TransformerDecoder(layers=fused_layer, num_layers=32, ...)
        >>>
        >>> # Original decoder state_dict still works
        >>> model.load_state_dict(..., strict=False)

    Args:
        layer (nn.Module): original decoder layer
        fusion_layer (nn.Module): new fusion layer
        fusion_first (bool): boolean to insert fusion layer before or after the decoder layer.
    """

    def __init__(self, layer: nn.Module, fusion_layer: nn.Module, fusion_first: bool = True):
        super().__init__()
        self.layer = layer
        self.fusion_layer = fusion_layer
        self.fusion_first = fusion_first

        # Keep FusionLayer wrappings out of the state_dict
        self._register_state_dict_hook(FusionLayer._state_dict_hook)
        self._register_load_state_dict_pre_hook(FusionLayer._load_state_dict_hook, with_module=True)
        # TODO: Switch to register_load_state_dict_pre_hook and
        # register_state_dict_pre_hook after PyTorch v2.5

    def _state_dict_hook(self, state_dict, prefix, *args, **kwargs):
        """Remove "layer" from the original layer in the state_dict
        name. This keeps the orginal state dict name for the layer
        from before fusing with the FusionLayer.

        [!Note] This update changes the order of the OrderedDict
        """
        keys = list(state_dict.keys())
        for key in keys:
            local_key = key[len(prefix) :]
            if local_key.startswith("layer"):
                new_key = prefix + local_key.replace("layer.", "")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    def _load_state_dict_hook(self, state_dict, prefix, *args, **kwargs):
        """Apply extra "layer" prefix to the state_dict key to
        account for the FusionLayer wrapping.
        """
        keys = list(state_dict.keys())
        for key in keys:
            local_key = key[len(prefix) :]
            if not local_key.startswith("fusion_layer"):
                new_key = prefix + "layer." + local_key
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        """Setup key value cache for both layers.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): maximum cache sequence length for cross-attention layer.
            decoder_max_seq_len (int): maximum cache sequence length for self-attention layer.
        """
        self.layer.setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )

        self.fusion_layer.setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup on ``self.layer``.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_setup`.
        """
        return self.layer.caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches on ``self.layer`` are enabled.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_enabled`.
        """
        return self.layer.caches_are_enabled()

    def reset_cache(self):
        """Reset both layers' key value caches."""
        self.layer.reset_cache()
        self.fusion_layer.reset_cache()

    def fusion_params(self) -> List[str]:
        """
        Return parameters of fusion layer.
        """
        fusion_params = [f"fusion_layer.{k}" for k, v in self.fusion_layer.named_parameters()]
        return fusion_params

    def forward(self, x: torch.Tensor, **kwargs: Dict) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            **kwargs (Dict): all additional layer args

        Returns
        -------
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]`

        """
        if self.fusion_first:
            x = self.fusion_layer(x, **kwargs)
            x = self.layer(x, **kwargs)
        else:
            x = self.layer(x, **kwargs)
            x = self.fusion_layer(x, **kwargs)
        return x


class FusionEmbedding(nn.Module):
    """Fusion embedding supports training additional special tokens while keeping
    the original embedding frozen. When fusing new models with a language model,
    there may be some additional tokens needed to support the fused language model. For
    example, adding a vision encoder might necessitate additional tokens like ``<|image|>``
    to indicate an images position in text and require learning an embedding for this token.
    The FusionEmbedding keeps the original embeddings frozen while learning a much smaller
    second embedding for the additional tokens. During forward this module routes
    the tokens to the appropriate embedding table.

    Use this as a drop-in replacement for :class:`torch.nn.Embedding` in your model.

    Example:
        >>> embedding = FusionEmbedding(vocab_size=100, fusion_vocab_size=10, embed_dim=128)
        >>> model = TransformerDecoder(tok_embeddings=embedding, ...)
        >>>
        >>> # Original model state_dict still works
        >>> model.load_state_dict(..., strict=False)

    .. note::
        This module assumes all tokens in the range [0, vocab_size) are part of the
        original embedding table and all new tokens in the range
        [vocab_size, vocab_size + fusion_vocab_size)

    Args:
        vocab_size (int): language model vocab size
        fusion_vocab_size (int): additional tokens for the fused model
        embed_dim (int): embedding dimension of the two embedding tables
    """

    def __init__(self, vocab_size: int, fusion_vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fusion_embedding = nn.Embedding(fusion_vocab_size, embed_dim)
        self.dim = embed_dim
        self.num_embeddings = vocab_size + fusion_vocab_size
        # TODO: Support merging the embeddings after finetuning

        # Keep FusionLayer wrappings out of the state_dict
        self._register_state_dict_hook(FusionEmbedding._state_dict_hook)
        self._register_load_state_dict_pre_hook(FusionEmbedding._load_state_dict_hook, with_module=True)
        # TODO: Switch to register_load_state_dict_pre_hook and
        # register_state_dict_pre_hook after PyTorch v2.5

    def _state_dict_hook(self, destination, prefix, keep_vars):
        """Remove "embedding" from the original embedding in the state_dict
        name. This keeps the orginal state dict name for the embedding
        from before fusing with the FusionEmbedding.

        [!Note] This update changes the order of the OrderedDict
        """
        key = prefix + "embedding.weight"
        new_key = prefix + "weight"
        destination[new_key] = destination[key]
        del destination[key]

    def _load_state_dict_hook(self, state_dict, prefix, *args, **kwargs):
        """Apply extra "embedding" prefix to the state_dict key to
        account for the FusionEmbedding wrapping.
        """
        if state_dict:
            key = prefix + "weight"
            new_key = prefix + "embedding.weight"
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    def fusion_params(self) -> List[str]:
        """
        Return fusion embedding parameters.
        """
        fusion_params = ["fusion_embedding.weight"]
        return fusion_params

    def _fused_embed(self, bs, seq_len):
        """
        Return an empty tensor the shape of the combined embedding.
        """
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        return torch.empty(bs, seq_len, self.dim, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input integer tensor with shape
                [batch_size x seq_length]

        Returns
        -------
            Tensor: output tensor embedding with shape
                [batch_size x seq_length x embed_dim]`

        """
        bs, seq_len = input.size()
        vocab_size = self.embedding.num_embeddings

        mask = input < vocab_size
        # num_tokens = (input < vocab_size).sum()
        tokens = torch.masked_select(input, mask)
        # num_fusion_tokens = (input >= vocab_size).sum()
        fusion_tokens = torch.masked_select(input, ~mask) - vocab_size

        # [batch_size x num_tokens x embed_dim]
        embeds = self.embedding(tokens)
        # [batch_size x num_fusion_tokens x embed_dim]
        fusion_embeds = self.fusion_embedding(fusion_tokens)

        # [batch_size x seq_length x embed_dim]
        out = self._fused_embed(bs, seq_len)
        mask = mask.unsqueeze(-1).expand(bs, seq_len, self.dim)
        out = out.masked_scatter(mask, embeds)
        out = out.masked_scatter(~mask, fusion_embeds)
        return out


def set_trainable_params(model: nn.Module, adapter_params: Dict[str, Any]) -> None:
    """
    Set trainable parameters for an nn.Module based on a state dict of adapter parameters.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.
        adapter_params (Dict[str, Any]): State dict mapping adapter key names to their
            respective nn.Parameters (i.e. outputs of :func:`~torchtune.modules.peft.get_adapter_params`.)

    Returns
    -------
        None
    """
    for k, v in model.named_parameters():
        v.requires_grad_(k in adapter_params)


class DeepFusionModel(nn.Module):
    """DeepFusion is a type of fused model architecture where a pretrained encoder is combined
    with a pretrained decoder (LLM). This is a popular architecture for multimodal models, with
    a full overview available in `The Evolution of Multimodal Model Architectures <https://arxiv.org/abs/2405.17927>`_.

    This module has the same methods and forward signature as :class:`~torchtune.modules.TransformerDecoder` and can be used
    interchangeably where :class:`~torchtune.modules.TransformerDecoder` is. It combines the encoder with the decoder as a
    single module for checkpointing and finetuning. It is expected that the encoder and decoder
    are already defined with any extra learnable ``fusion_params``: learnable parameters to help
    adapt the pre-trained encoder to the pre-trained decoder.

    Example:
        >>> # decoder is a TransformerDecoder (e.g. llama3_8b) with fused cross attention layers
        >>> embed = FusionEmbedding(...)
        >>> layer = FusionLayer(
        ...     layer=TransformerSelfAttentionLayer(...),
        ...     fusion_layer=TransformerCrossAttentionLayer(...),
        ... )
        >>> decoder = TransformerDecoder(tok_embeddings=embed, layers=layer, num_layers=32, ...)
        >>>
        >>> # encoder is pre-trained encoder (e.g. clip_vit_224) with an added projection head
        >>> projection_head = FeedForward(...)
        >>> register_fusion_module(projection_head))
        >>> encoder = nn.Sequential(clip_vit_224(), projection_head)
        >>>
        >>> # DeepFusionModel combines the encoder and decoder
        >>> model = DeepFusionModel(decoder, encoder)
        >>>
        >>> # Load full fused checkpoints (e.g. a Flamingo checkpoint)
        >>> model.load_state_dict(...)
        >>>
        >>> # Or load pretrained individual models (fusion_params are not loaded)
        >>> model.encoder.load_state_dict(..., strict=False)
        >>> model.decoder.load_state_dict(..., strict=False)
        >>>
        >>> # Forward pass
        >>> output = model(tokens, mask, encoder_input, encoder_mask, input_pos)

    Args:
        decoder (TransformerDecoder): decoder module
        encoder (nn.Module): encoder module
        decoder_trainable (bool): whether to train or freeze the decoder. Default is False.
        encoder_trainable (bool): whether to train or freeze the encoder. Default is False.
        fusion_trainable (bool): whether to train the fusion parameters. Default is True.

    """

    def __init__(
        self,
        decoder: TransformerDecoder,
        encoder: nn.Module,
        *,
        decoder_trainable: bool = False,
        encoder_trainable: bool = False,
        fusion_trainable: bool = True,
    ):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

        trainable_params = set()
        if encoder_trainable:
            trainable_params |= {f"encoder.{n}" for n, p in self.encoder.named_parameters()}
        if decoder_trainable:
            trainable_params |= {f"decoder.{n}" for n, p in self.decoder.named_parameters()}
        if fusion_trainable:
            trainable_params |= set(get_fusion_params(self))
        else:
            trainable_params -= set(get_fusion_params(self))
        set_trainable_params(self, trainable_params)

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe.
        """
        self.decoder.set_num_output_chunks(num_output_chunks)

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int = None,
        decoder_max_seq_len: int = None,
    ):
        """
        Sets up key-value attention caches for inference for ``self.decoder``.
        For each layer in ``self.decoder.layers``:
        - :class:`torchtune.modules.TransformerSelfAttentionLayer` will use ``decoder_max_seq_len``.
        - :class:`torchtune.modules.TransformerCrossAttentionLayer` will use ``encoder_max_seq_len``.
        - :class:`torchtune.modules.fusion.FusionLayer` will use both ``decoder_max_seq_len`` and ``encoder_max_seq_len``.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): maximum encoder cache sequence length.
            decoder_max_seq_len (int): maximum decoder cache sequence length.
        """
        self.decoder.setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.decoder.caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`~torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.decoder.caches_are_enabled()

    def reset_caches(self):
        """
        Resets KV-cache buffers on relevant attention modules to zero, and reset cache positions to zero,
        without deleting or reallocating cache tensors.
        """
        self.decoder.reset_caches()

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[Dict] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            mask (Optional[torch.Tensor]): Optional boolean tensor which contains the attention mask
                with shape ``[b x s x s]``. This is applied after the query-key multiplication and
                before the softmax. A value of True in row i and column j means token i attends
                to token j. A value of False means token i does not attend to token j. If no
                mask is specified, a causal mask is used by default. Default is None.
            encoder_input (Optional[Dict]): Optional input for the encoder.
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position i,j means token i can attend
                to embedding j in the decoder. Mask has shape ``[b x s x s_e]``. Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns
        -------
            Tensor: output tensor with shape ``[b x s x v]`` or a list of layer \
                output tensors defined by ``output_hidden_states`` with the \
                final output tensor appended to the list.

        Notation used for tensor shapes:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # During decoding, encoder_input will only be provided
        # for new inputs. Previous encoder outputs are cached
        # in the decoder cache.
        encoder_embed = None
        if encoder_input is not None:
            encoder_embed = self.encoder(**encoder_input)

        output = self.decoder(
            tokens=tokens,
            mask=mask,
            encoder_input=encoder_embed,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )
        return output


class Llama3VisionProjectionHead(nn.Module):
    """Projection transformer to adapt the output of a
    pretrained frozen encoder (CLIP) to a pretrained decoder model.
    For example, nn.Sequential(CLIP(), Llama3VisionProjectionHead()).

    Args:
        layers (nn.Module): Transformer Decoder layers
        output (nn.Module): Output linear layer. Input dim is
            (num_hidden + 1) * encoder_dim and output is decoder_dim.
        num_hidden_inputs (int): Number of expected hidden state inputs
    """

    def __init__(
        self,
        layers: nn.Module,
        output: nn.Module,
        num_hidden_inputs: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.output = output
        self.num_hidden = num_hidden_inputs

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x i x t x e x d]
            hidden_states (Optional[List[torch.Tensor]]): list of hidden states
                from the encoder. Each hidden state has the same shape as x.

        Returns
        -------
            Tensor: output tensor of a sequence of embedings [b x s x d]
                where sequence length is num_imgs*num_tiles+num_embeds

        Notation used for tensor shapes:
            - b: batch size
            - i: number of images
            - t: number of tiles (where a single image is broken into multiple tiles)
            - e: number of embeds per tile (e.g. CLS embed + patch embeds, etc.)
            - s: sequence length computed by i*t*e
            - d: embed dim
        """
        bsz, imgs, tiles, embeds, dim = x.shape

        # apply transformer layers
        x = x.view(bsz * imgs, tiles * embeds, dim)
        for layer in self.layers:
            x = layer(x)
        x = x.view(bsz, imgs, tiles, embeds, dim)

        # interleave hidden states and cat with x
        if self.num_hidden > 0:
            hidden_states = torch.stack(hidden_states, dim=-1)
            hidden_states = hidden_states.view(bsz, imgs, tiles, embeds, -1)
            x = torch.cat([x, hidden_states], dim=-1)

        # shape [b x s x d]
        x = self.output(x).reshape(bsz, imgs * tiles * embeds, -1)

        return x


class Llama3VisionEncoder(nn.Module):
    """Vision encoder model for Llama 3.2 Vision. This combines a pretrained
    vision encoder with a learnable projection head. The projection head
    is converted to a fusion module and supports fusion utils.

    Args:
        clip (nn.Module): CLIP encoder vision model
        projection_head (nn.Module): projection_head that takes embeddings
            with dimension encoder_dim as input and outputs embeddings of
            size decoder_dim.
    """

    def __init__(self, clip: nn.Module, projection_head: nn.Module) -> None:
        super().__init__()
        self.clip = clip
        self.projection = projection_head
        register_fusion_module(self.projection)

    def forward(self, images: torch.Tensor, aspect_ratio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Image tensor with shape [b x i x t x c x w x h]
            aspect_ratio (Optional[torch.Tensor]): Tensor with shape [b x i x 2]. If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.

        Returns
        -------
            Tensor: output tensor of a sequence of embedings [b x s x d]
                where sequence length is num_imgs*num_tiles+num_embeds

         Notation used for tensor shapes:
            - b: batch size
            - i: number of images
            - t: number of tiles (where a single image is broken into multiple tiles)
            - c: number of image channels (e.g. rgb = 3)
            - w: image width
            - h: image height
            - s: sequence length computed by i*t*clip_embeds_per_tile
            - d: embed dim
        """
        x, hidden_states = self.clip(images, aspect_ratio)
        x = self.projection(x, hidden_states)
        return x


def llama3_2_vision_projection_head(
    *,
    num_layers: int,
    num_heads: int,
    decoder_embed_dim: int,
    clip_embed_dim: int,
    num_hidden_inputs: int,
) -> Llama3VisionProjectionHead:
    """
    Build the Llama 3.2 Vision Projection Head that maps the output of the CLIP encoder
    to the decoder cross attention input.

    Args:
        num_layers (int): number of layers in the projection head.
        num_heads (int): number of heads in the projection head.
        decoder_embed_dim (int): embedding dimension for the decoder.
        clip_embed_dim (int): embedding dimension for the CLIP encoder.
        num_hidden_inputs (int): number of hidden inputs to the projection head.

    Returns
    -------
        Llama3VisionProjectionHead: Instantiation of Llama 3.2 vision projection head.
    """
    mlp_ratio = 4
    hidden_dim = int(mlp_ratio * clip_embed_dim)
    head_dim = clip_embed_dim // num_heads
    num_kv_heads = num_heads

    layers = []
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=clip_embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(clip_embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(clip_embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(clip_embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(clip_embed_dim, clip_embed_dim, bias=False),
            pos_embeddings=None,
            attn_dropout=0.0,
            is_causal=False,
        )

        mlp = clip_mlp(
            in_dim=clip_embed_dim,
            hidden_dim=hidden_dim,
            out_dim=clip_embed_dim,
            activation=nn.GELU(),
        )

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=Fp32LayerNorm(clip_embed_dim, eps=1e-5),
            mlp_norm=Fp32LayerNorm(clip_embed_dim, eps=1e-5),
            sa_scale=TanhGate(),
            mlp_scale=TanhGate(),
        )
        layers.append(layer)

    # we concatenate clip embeddings and hidden layers output
    # and project it to embed_dim_out, which will be used for the
    # cross encoding
    proj_in = clip_embed_dim * (num_hidden_inputs + 1)
    return Llama3VisionProjectionHead(
        layers=layers,
        output=nn.Linear(proj_in, decoder_embed_dim),
        num_hidden_inputs=num_hidden_inputs,
    )


def llama3_2_vision_encoder(
    # clip encoder parameters
    *,
    patch_size: int,
    num_heads: int,
    clip_embed_dim: int,
    clip_num_layers: int,
    clip_hidden_states: Optional[List[int]],
    # projection parameters
    num_layers_projection: int,
    decoder_embed_dim: int,
    # image parameters
    tile_size: int,
    max_num_tiles: int = 4,
    in_channels: int = 3,
) -> Llama3VisionEncoder:
    """
    Build the Llama 3.2 vision encoder by combining the CLIP image model with an additional
    projection head fusion module. This includes:
    - Spatial positional encodings
    - CLIP model backbone
    - Projection head on top of CLIP
    - Final projection into token embedding dimension

    Args:
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        num_heads (int): The number of attention heads in each transformer layer.
        clip_embed_dim (int): The dimensionality of each patch embedding in CLIP.
        clip_num_layers (int): The number of transformer layers.
        clip_hidden_states (Optional[List[int]]): The indices of CLIP hidden layers to return
            to return to the encoder projection head. It will return the intermediate results
            of the vision transformer layers which will be concatenated with the CLIP output
            and input into the projection head. For example, ``clip_hidden_states=[0,3]`` will
            return the embeddings before they go through the first and fourth layers.
        num_layers_projection (int): The number of transformer layers in the projection head.
        decoder_embed_dim (int): The dimensionality of the final output embeddings for the decoder.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings.
        in_channels (int): The number of image input channels.

    Returns
    -------
        Llama3VisionEncoder: Instantiation of Llama 3.2 vision encoder.
    """
    # clip encoder
    clip = clip_vision_encoder(
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=clip_embed_dim,
        num_layers=clip_num_layers,
        num_heads=num_heads,
        activation=nn.GELU,
        out_indices=clip_hidden_states,
        max_num_tiles=max_num_tiles,
        in_channels=in_channels,
        attn_bias=False,
        output_cls_projection=False,
    )

    # Projection head
    projection_head = llama3_2_vision_projection_head(
        num_layers=num_layers_projection,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        clip_embed_dim=clip_embed_dim,
        num_hidden_inputs=len(clip_hidden_states or []),
    )

    return Llama3VisionEncoder(clip=clip, projection_head=projection_head)


def llama3_2_vision_decoder(
    *,
    vocab_size: int,
    num_layers: int,
    fusion_interval: int,
    num_special_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    encoder_max_seq_len: int,
    rope_base: int = 500000.0,
    intermediate_dim: Optional[int] = None,
) -> TransformerDecoder:
    """
    Build the decoder associated with the Llama3 model with additional fused
    cross attention layers. This includes:
    - Token embeddings
    - num_layers number of CausalSelfAttention blocks
    - Fused cross attention layers every fusion_interval number of layers
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        fusion_interval (int): interval number of layers between fusion layers.
        num_special_tokens (int): number of special tokens added for the fusion model.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value.
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`.
        encoder_max_seq_len (int): maximum sequence length the encoder will be run with, as used
            by :func:`~torchtune.modules.KVCache`.
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`.

    Returns
    -------
        TransformerDecoder: Instantiation of Llama 3.2 vision decoder.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    hidden_dim = intermediate_dim or scale_hidden_dim_for_mlp(embed_dim)
    layers = []

    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)

    for idx in range(1, num_layers + 1):
        # Self attention layers for text decoder
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=0.0,
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        decoder_layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=1e-5),
            mlp_norm=RMSNorm(dim=embed_dim, eps=1e-5),
        )

        # cross attention layers, mixing text and vision,
        # placed every `fusion_interval` layers
        if idx % fusion_interval == 0:
            attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                q_norm=RMSNorm(dim=head_dim, eps=1e-05),
                k_norm=RMSNorm(dim=head_dim, eps=1e-05),
                pos_embeddings=None,
                max_seq_len=encoder_max_seq_len,
                is_causal=False,
                attn_dropout=0.0,
            )
            mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
            xattn_layer = TransformerCrossAttentionLayer(
                attn=attn,
                mlp=mlp,
                ca_norm=RMSNorm(dim=embed_dim),
                mlp_norm=RMSNorm(dim=embed_dim),
                ca_scale=TanhGate(),
                mlp_scale=TanhGate(),
            )
            fusion_layer = FusionLayer(layer=decoder_layer, fusion_layer=xattn_layer)
            layers.append(fusion_layer)
        else:
            layers.append(decoder_layer)

    tok_embeddings = FusionEmbedding(vocab_size, num_special_tokens, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=1e-05),
        output=output_proj,
    )


def llama3_2_vision_transform(
    path: str,
    max_seq_len: int = 8192,
    image_size: int = 560,
    special_tokens_path: Optional[str] = None,
    prompt_template: Optional[_TemplateType] = None,
) -> Llama3VisionTransform:
    """
    Data Transforms (including Tokenizer) for Llama3 Vision.

    Args:
        path (str): path to the tokenizer
        max_seq_len (int): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated.
        image_size (int): Base image size that images will be tiled and resized to.
            Default is 560 for Instruct weights, use 448 for pre-trained.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Llama3 special tokens.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.

    Returns
    -------
        Llama3VisionTransform: Instantiation of the Llama 3.2 vision transform
    """
    special_tokens = parse_hf_tokenizer_json(special_tokens_path) if special_tokens_path is not None else None
    template = _get_prompt_template(prompt_template) if prompt_template is not None else None
    return Llama3VisionTransform(
        path=path,
        special_tokens=special_tokens,
        tile_size=image_size,
        patch_size=14,
        max_num_tiles=4,
        max_seq_len=max_seq_len,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        prompt_template=template,
    )


# -----------------------------------------------------------------------------

# # temp for format


@dataclass
class Llama_32_11B_VisionConfig:
    decoder_trainable: bool = False
    encoder_trainable: bool = True
    fusion_trainable: bool = True
    image_size: int = 560

    patch_size = 14
    num_heads = 16
    clip_embed_dim = 1280
    clip_num_layers = 32
    clip_hidden_states = [3, 7, 15, 23, 30]
    decoder_embed_dim = 4096
    num_layers_projection = 8
    tile_size = image_size
    max_num_tiles = 4
    in_channels = 3

    vocab_size = 128256  # number of tokens: 127,999 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    num_layers = 16  # TODO: 32
    fusion_interval = 4
    num_special_tokens = 8
    num_heads = 32
    num_kv_heads = 8
    max_seq_len = 131072
    encoder_max_seq_len = 128080  # 20*6404
    rope_base = 500000.0
    intermediate_dim = 14336


class Llama_32_11B_Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = llama3_2_vision_encoder(
            patch_size=self.config.patch_size,
            num_heads=self.config.num_heads,
            clip_embed_dim=self.config.clip_embed_dim,
            clip_num_layers=self.config.clip_num_layers,
            clip_hidden_states=self.config.clip_hidden_states,
            decoder_embed_dim=self.config.decoder_embed_dim,
            num_layers_projection=self.config.num_layers_projection,
            tile_size=self.config.image_size,
            max_num_tiles=self.config.max_num_tiles,
            in_channels=self.config.in_channels,
        )
        self.decoder = llama3_2_vision_decoder(
            vocab_size=self.config.vocab_size,
            num_layers=self.config.num_layers,
            fusion_interval=self.config.fusion_interval,
            num_special_tokens=self.config.num_special_tokens,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            embed_dim=self.config.decoder_embed_dim,
            max_seq_len=self.config.max_seq_len,
            encoder_max_seq_len=self.config.encoder_max_seq_len,
            rope_base=self.config.rope_base,
            intermediate_dim=self.config.intermediate_dim,
        )
        self.model = DeepFusionModel(
            encoder=self.encoder,
            decoder=self.decoder,
            encoder_trainable=self.config.encoder_trainable,
            decoder_trainable=self.config.decoder_trainable,
            fusion_trainable=self.config.fusion_trainable,
        )

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
    def from_pretrained(  # noqa: C901
        cls,
    ):
        """
        Loads pretrained Llama-3.2-11B-Vision model weights from huggingface.
        - Updating the cross attention layer numbers
        - skip loading the rope embeddings
        - reshaping q, k projections
        - reversing the precomputed vision positional embeddings
        """
        from transformers import MllamaForConditionalGeneration

        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        print(f"loading weights from pretrained model: {model_id}")

        # # create a from-scratch initialized model
        config = Llama_32_11B_VisionConfig()
        model = Llama_32_11B_Vision(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        sd_hf = model_hf.state_dict()

        converted_state_dict = {}
        head_dim = config.decoder_embed_dim // config.num_heads
        cross_attention_layers = []

        def _permute(t, n_heads):
            return (
                t.view(n_heads, 2, head_dim // 2, config.decoder_embed_dim)
                .transpose(1, 2)
                .reshape((head_dim * n_heads), config.decoder_embed_dim)
            )

        _FROM_HF = {
            "language_model.model.embed_tokens.weight": "decoder.tok_embeddings.weight",
            "language_model.model.layers.{}.self_attn.q_proj.weight": "decoder.layers.{}.attn.q_proj.weight",
            "language_model.model.layers.{}.self_attn.k_proj.weight": "decoder.layers.{}.attn.k_proj.weight",
            "language_model.model.layers.{}.self_attn.v_proj.weight": "decoder.layers.{}.attn.v_proj.weight",
            "language_model.model.layers.{}.self_attn.o_proj.weight": "decoder.layers.{}.attn.output_proj.weight",
            "language_model.model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "language_model.model.layers.{}.mlp.gate_proj.weight": "decoder.layers.{}.mlp.w1.weight",
            "language_model.model.layers.{}.mlp.up_proj.weight": "decoder.layers.{}.mlp.w3.weight",
            "language_model.model.layers.{}.mlp.down_proj.weight": "decoder.layers.{}.mlp.w2.weight",
            "language_model.model.layers.{}.input_layernorm.weight": "decoder.layers.{}.sa_norm.scale",
            "language_model.model.layers.{}.post_attention_layernorm.weight": "decoder.layers.{}.mlp_norm.scale",
            "language_model.model.norm.weight": "decoder.norm.scale",
            "language_model.lm_head.weight": "decoder.output.weight",
            "language_model.model.layers.{}.cross_attn_attn_gate": "decoder.layers.{}.fusion_layer.ca_scale.scale",
            "language_model.model.layers.{}.cross_attn_mlp_gate": "decoder.layers.{}.fusion_layer.mlp_scale.scale",
            "language_model.model.layers.{}.cross_attn.q_proj.weight": "decoder.layers.{}.fusion_layer.attn.q_proj.weight",
            "language_model.model.layers.{}.cross_attn.k_proj.weight": "decoder.layers.{}.fusion_layer.attn.k_proj.weight",
            "language_model.model.layers.{}.cross_attn.v_proj.weight": "decoder.layers.{}.fusion_layer.attn.v_proj.weight",
            "language_model.model.layers.{}.cross_attn.o_proj.weight": "decoder.layers.{}.fusion_layer.attn.output_proj.weight",
            "language_model.model.layers.{}.cross_attn.q_norm.weight": "decoder.layers.{}.fusion_layer.attn.q_norm.scale",
            "language_model.model.layers.{}.cross_attn.k_norm.weight": "decoder.layers.{}.fusion_layer.attn.k_norm.scale",
            "vision_model.gated_positional_embedding.embedding": "encoder.clip.token_pos_embedding.local_token_positional_embedding",
            "vision_model.gated_positional_embedding.tile_embedding.weight": "encoder.clip.token_pos_embedding.global_token_positional_embedding",  # noqa
            "vision_model.gated_positional_embedding.gate": "encoder.clip.token_pos_embedding.gate",
            "vision_model.layernorm_pre.weight": "encoder.clip.ln_pre.weight",
            "vision_model.layernorm_pre.bias": "encoder.clip.ln_pre.bias",
            "vision_model.layernorm_post.weight": "encoder.clip.ln_post.weight",
            "vision_model.layernorm_post.bias": "encoder.clip.ln_post.bias",
            "vision_model.pre_tile_positional_embedding.embedding.weight": "encoder.clip.pre_tile_pos_embed.embedding",
            "vision_model.pre_tile_positional_embedding.gate": "encoder.clip.pre_tile_pos_embed.gate",
            "vision_model.post_tile_positional_embedding.embedding.weight": "encoder.clip.post_tile_pos_embed.embedding",
            "vision_model.post_tile_positional_embedding.gate": "encoder.clip.post_tile_pos_embed.gate",
            "vision_model.class_embedding": "encoder.clip.cls_token_embedding.weight",
            "vision_model.patch_embedding.weight": "encoder.clip.conv.weight",
            "vision_model.transformer.layers.{}.self_attn.q_proj.weight": "encoder.clip.layers.{}.attn.q_proj.weight",
            "vision_model.transformer.layers.{}.self_attn.k_proj.weight": "encoder.clip.layers.{}.attn.k_proj.weight",
            "vision_model.transformer.layers.{}.self_attn.v_proj.weight": "encoder.clip.layers.{}.attn.v_proj.weight",
            "vision_model.transformer.layers.{}.self_attn.o_proj.weight": "encoder.clip.layers.{}.attn.output_proj.weight",
            "vision_model.transformer.layers.{}.mlp.fc1.weight": "encoder.clip.layers.{}.mlp.w1.weight",
            "vision_model.transformer.layers.{}.mlp.fc1.bias": "encoder.clip.layers.{}.mlp.w1.bias",
            "vision_model.transformer.layers.{}.mlp.fc2.weight": "encoder.clip.layers.{}.mlp.w2.weight",
            "vision_model.transformer.layers.{}.mlp.fc2.bias": "encoder.clip.layers.{}.mlp.w2.bias",
            "vision_model.transformer.layers.{}.input_layernorm.weight": "encoder.clip.layers.{}.sa_norm.weight",
            "vision_model.transformer.layers.{}.input_layernorm.bias": "encoder.clip.layers.{}.sa_norm.bias",
            "vision_model.transformer.layers.{}.post_attention_layernorm.weight": "encoder.clip.layers.{}.mlp_norm.weight",
            "vision_model.transformer.layers.{}.post_attention_layernorm.bias": "encoder.clip.layers.{}.mlp_norm.bias",
            "vision_model.global_transformer.layers.{}.self_attn.q_proj.weight": "encoder.projection.layers.{}.attn.q_proj.weight",
            "vision_model.global_transformer.layers.{}.self_attn.k_proj.weight": "encoder.projection.layers.{}.attn.k_proj.weight",
            "vision_model.global_transformer.layers.{}.self_attn.v_proj.weight": "encoder.projection.layers.{}.attn.v_proj.weight",
            "vision_model.global_transformer.layers.{}.self_attn.o_proj.weight": "encoder.projection.layers.{}.attn.output_proj.weight",
            "vision_model.global_transformer.layers.{}.mlp.fc1.weight": "encoder.projection.layers.{}.mlp.w1.weight",
            "vision_model.global_transformer.layers.{}.mlp.fc1.bias": "encoder.projection.layers.{}.mlp.w1.bias",
            "vision_model.global_transformer.layers.{}.mlp.fc2.weight": "encoder.projection.layers.{}.mlp.w2.weight",
            "vision_model.global_transformer.layers.{}.mlp.fc2.bias": "encoder.projection.layers.{}.mlp.w2.bias",
            "vision_model.global_transformer.layers.{}.input_layernorm.weight": "encoder.projection.layers.{}.sa_norm.weight",
            "vision_model.global_transformer.layers.{}.input_layernorm.bias": "encoder.projection.layers.{}.sa_norm.bias",
            "vision_model.global_transformer.layers.{}.post_attention_layernorm.weight": "encoder.projection.layers.{}.mlp_norm.weight",
            "vision_model.global_transformer.layers.{}.post_attention_layernorm.bias": "encoder.projection.layers.{}.mlp_norm.bias",
            "vision_model.global_transformer.layers.{}.gate_attn": "encoder.projection.layers.{}.sa_scale.scale",
            "vision_model.global_transformer.layers.{}.gate_ffn": "encoder.projection.layers.{}.mlp_scale.scale",
            "multi_modal_projector.weight": "encoder.projection.output.weight",
            "multi_modal_projector.bias": "encoder.projection.output.bias",
        }

        def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
            try:
                # Checks if there is a layer # in the key
                if any(k.isdigit() for k in key.split(".")):
                    # Replace layer number with "{}" to create key for lookup
                    abstract_key = re.sub(r"(\.\d+)", ".{}", key)
                    layer_num = re.search(r"\d+", key).group(0)
                    new_key = mapping_dict[abstract_key]
                    new_key = new_key.format(layer_num)
                else:
                    new_key = mapping_dict[key]
            except KeyError as e:
                raise Exception(
                    f'Error converting the state dict. Found unexpected key: "{key}". '
                    "Please make sure you're loading a checkpoint with the right format. "
                ) from e

            return new_key

        for key, value in sd_hf.items():
            if "rotary_emb.inv_freq" in key:  # Skip loading the position embeddings
                continue
            new_key = get_mapped_key(key, _FROM_HF)
            if "language_model" in key:
                if "layers" in key:  # Update layer numbers
                    layer = int(key.split(".")[3])
                    num_shifts = sum(layer > la for la in cross_attention_layers)
                    new_layer = layer - num_shifts
                    key_lst = new_key.split(".")
                    if layer in cross_attention_layers and "fusion_layer" not in new_key:
                        # some keys are the same for sa and ca, so we need to edit them here
                        key_lst[2] = f"{new_layer}.fusion_layer"
                        if "sa_norm" in new_key:
                            key_lst[3] = "ca_norm"
                    else:
                        key_lst[2] = str(new_layer)
                    new_key = ".".join(key_lst)
                if "q_proj" in key and "cross_attn" not in key:
                    value = _permute(value, config.num_heads)
                elif "k_proj" in key and "cross_attn" not in key:
                    value = _permute(value, config.num_kv_heads)
                elif new_key == "decoder.tok_embeddings.weight":
                    # Split embedding between learnable embeddings and original text embedding
                    learned_embedding = "decoder.tok_embeddings.fusion_embedding.weight"
                    converted_state_dict[learned_embedding] = value[config.vocab_size :]
                    value = value[: config.vocab_size]
            elif "vision_model" in key:
                if "tile_pos_embed.embedding" in new_key or "global_token_positional_embedding" in new_key:
                    # WARNING: META format postional embeddings contain embeddings that
                    # the model can never use (4 tiles -> 4 x 4 embeddings -> a 4 x 4 image would be 16 tiles).
                    # HF removes these extra embeddings, for us to convert to the META format we set those
                    # unused embeddings as 0 instead of the original random (untrained) values in the original
                    # META checkpoing
                    num_embeds = value.shape[-1] // config.clip_embed_dim // config.max_num_tiles
                    pos_embedding = torch.zeros(
                        config.max_num_tiles,
                        config.max_num_tiles,
                        num_embeds,
                        config.clip_embed_dim,
                        device=value.device,
                        dtype=value.dtype,
                    )
                    # Loop through aspect ratios and assign precomputed embeds back to Meta Llama embeddings
                    for i, (h, w) in enumerate([]):
                        if h * w == config.max_num_tiles:  # h*w < num_tiles is redundant
                            # i == 0 is used for padding in HF
                            pos_embedding[:h, :w] = value[i + 1].reshape(h, w, num_embeds, config.clip_embed_dim)
                    value = pos_embedding

            converted_state_dict[new_key] = value

        # Load the converted state dict
        sd.update(converted_state_dict)
        return model

    # def configure_optimizers(self, weight_decay, learning_rate, device_type):
    #     # start with all of the candidate parameters (that require grad)
    #     param_dict = dict(self.named_parameters())
    #     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    #     # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    #     # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    #     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    #     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    #     optim_groups = [
    #         {"params": decay_params, "weight_decay": weight_decay},
    #         {"params": nodecay_params, "weight_decay": 0.0},
    #     ]
    #     # Create AdamW optimizer and use the fused version if it is available
    #     fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    #     use_fused = fused_available and device_type == "cuda"
    #     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    #     return optimizer


# def llama3_vision_tune_to_hf(
#     state_dict: Dict[str, torch.Tensor],
#     num_heads: int = 32,
#     num_kv_heads: int = 32,
#     dim: int = 4096,
#     head_dim: int = None,
#     vocab_size: int = 128256,
#     cross_attention_layers: Optional[List[int]] = None,
#     # Vision Encoder Paramters
#     encoder_dim: int = 1280,
#     tile_size: int = 448,
#     num_tiles: int = 4,
#     supported_aspect_ratios: List[Tuple[int, int]] = None,
# ) -> Dict[str, torch.Tensor]:
#     """
#     Convertor from Tune state dict to HF state dict. This handles:
#     - Updateing the cross attention layer numbers
#     - skip loading the rope embeddings
#     - reshaping q, k projections
#     """
#     converted_state_dict = {}
#     inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}
#     # missing keys in _FROM_HF due to naming collisions
#     missing_keys = {
#         "decoder.layers.{}.fusion_layer.ca_norm.scale": "language_model.model.layers.{}.input_layernorm.weight",
#         "decoder.layers.{}.fusion_layer.mlp_norm.scale": "language_model.model.layers.{}.post_attention_layernorm.weight",
#         "decoder.layers.{}.fusion_layer.mlp.w1.weight": "language_model.model.layers.{}.mlp.gate_proj.weight",
#         "decoder.layers.{}.fusion_layer.mlp.w3.weight": "language_model.model.layers.{}.mlp.up_proj.weight",
#         "decoder.layers.{}.fusion_layer.mlp.w2.weight": "language_model.model.layers.{}.mlp.down_proj.weight",
#         "decoder.tok_embeddings.fusion_embedding.weight": None,
#     }
#     inverted_mapping_dict.update(missing_keys)

#     if head_dim is None:
#         head_dim = dim // num_heads
#     if cross_attention_layers is None:
#         cross_attention_layers = []
#     # convert hf layer numbers to tune numbers
#     cross_attention_layers = [l - i for i, l in enumerate(sorted(cross_attention_layers))]

#     def _permute(t, n_heads):
#         return t.view(n_heads, head_dim // 2, 2, dim).transpose(1, 2).reshape((head_dim * n_heads), dim)

#     for key, value in state_dict.items():
#         new_key = get_mapped_key(key, inverted_mapping_dict)
#         if "decoder" in key:
#             if "layers" in key:  # Update layer numbers
#                 layer = int(key.split(".")[2])
#                 num_shifts = sum(layer > l for l in cross_attention_layers)
#                 new_layer = layer + num_shifts
#                 key_lst = new_key.split(".")
#                 if layer in cross_attention_layers and "fusion_layer" not in key:
#                     new_layer += 1  # hf treats the fusion_layer as an additional layer
#                 key_lst[3] = str(new_layer)
#                 new_key = ".".join(key_lst)
#             if "q_proj" in key and "cross_attn" not in new_key:
#                 value = _permute(value, num_heads)
#             elif "k_proj" in key and "cross_attn" not in new_key:
#                 value = _permute(value, num_kv_heads)
#             elif key == "decoder.tok_embeddings.weight":
#                 learned_embedding = state_dict["decoder.tok_embeddings.fusion_embedding.weight"]
#                 value = torch.cat([value, learned_embedding])
#             elif key == "decoder.tok_embeddings.fusion_embedding.weight":
#                 continue
#         elif "encoder" in key:
#             if "tile_pos_embed.embedding" in key or "global_token_positional_embedding" in key:
#                 num_embeds = value.shape[-2]
#                 pos_embedding = torch.zeros(
#                     len(supported_aspect_ratios) + 1,
#                     num_tiles,
#                     num_embeds,
#                     encoder_dim,
#                     device=value.device,
#                     dtype=value.dtype,
#                 )
#                 # Loop through aspect ratios and precompute embeds per aspect ratio
#                 for i, (h, w) in enumerate(supported_aspect_ratios or []):
#                     pos_embedding[i + 1, : h * w] = value[:h, :w].reshape(h * w, num_embeds, encoder_dim)
#                 value = pos_embedding.flatten(1)

#         converted_state_dict[new_key] = value
#     return converted_state_dict


llama_32_11b_vision = Llama_32_11B_Vision.from_pretrained()
print("Loaded Llama-3.2-11B-Vision model")
