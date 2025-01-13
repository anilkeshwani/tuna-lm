# based on the torchtune library:
# from torchtune.data import InputOutputToMessages

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Union

import numpy as np
from datasets import load_dataset
from sardalign.utils import dsu2pua
from torch.utils.data import Dataset
from torchtune.data import Message, PromptTemplate, Role
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import validate_messages
from torchtune.data._utils import format_content_with_images, load_image
from torchtune.datasets._packed import PackedDataset
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


# NOTE we replace torchtune.datasets._sft.SFTDataset with a custom impl to support the inference use case
# NOTE we also rename the model_transform to tokenizer, since that's what it is in our case
# from torchtune.datasets._sft import SFTDataset # commented out (torchtune impl. use previously)


class SFTDataset(Dataset):
    """
    Primary class for creating any dataset for supervised fine-tuning either from
    Hugging Face Hub, local files, or remote files. This class supports instruct,
    chat, tool, or multimodal data for fine-tuning. At a high level, this class
    will load the data from source and apply the following pre-processing steps
    when a sample is retrieved:

    1. Dataset-specific transform. This is typically unique to each dataset and extracts
       the necessary columns into torchtune's :class:`~torchtune.data.Message` format,
       a standardized API for all model tokenizers.
    2. Model-specific transform or tokenization with optional prompt template


    All datasets are formatted into a list of :class:`~torchtune.data.Message`
    because for fine-tuning, datasets can be considered as "conversations" with the model,
    or AI assistant. Thus, we can standardize all text content as messages in a conversation assigned to
    a role:

    - ``"system"`` messages contain the system prompt
    - ``"user"`` messages contain the input prompt into the model
    - ``"assistant"`` messages are the response of the model and what you actually want
      to train for and compute loss directly against
    - ``"ipython"`` messages are the return from a tool call

    Chat datasets are multiple rounds of user-assistant messages. Instruct datasets
    are typically a single round involving a specific instruction and the model's response.
    Tool datasets are a type of chat dataset that includes ipython messages. Multimodal
    datasets are a type of chat dataset that incorporates media into the user messages.

    The :class:`~torchtune.data.Message` forms the core data unit that all tokenizer
    APIs expect. The key component of this class that ensures any dataset is transformed
    into this format is the ``message_transform``. This is a callable class that takes
    in a sample dictionary - typically a single row from the source dataset - that
    processes the sample in any configurable way to output a list of messages::

        [
            Message(
                role=<system|user|assistant|ipython>,
                content=<message>,
            ),
            ...
        ]

    For any custom dataset, use the ``message_transform`` to contain all pre-processing to
    return the list of messages.

    Legacy documentation on model_transform: Any model-specific pre-processing that needs to happen
    can be configured with the ``model_transform``
    parameter. This is another callable class that contains any custom logic tied to the
    model you are fine-tuning and will carry over to inference. For example, text + image
    multimodal datasets requires processing the images in a way specific to the vision
    encoder being used by the model and is agnostic to the specific dataset.

    Legacy documentation on model_transform: Tokenization is handled by the ``model_transform``.
    All :class:`~torchtune.modules.tokenizers.ModelTokenizer`
    can be treated as a ``model_transform`` since it uses the model-specific tokenizer to
    transform the list of messages outputted from the ``message_transform`` into tokens
    used by the model for training. Text-only datasets will simply pass the
    :class:`~torchtune.modules.tokenizers.ModelTokenizer` into ``model_transform``.
    Tokenizers handle prompt templating, if configured.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        message_transform (Transform): callable that keys into the desired fields in the sample
            and converts text content to a list of :class:`~torchtune.data.Message`. It is expected that the final list
            of messages are stored in the ``"messages"`` key.
        model_transform (Transform): callable that applies model-specific pre-processing to the sample after the list of
            messages is created from ``message_transform``. This includes tokenization and any modality-specific
            transforms. It is expected to return at minimum ``"tokens"`` and ``"mask"`` keys.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.
    """

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        tokenizer: Llama3Tokenizer,
        filter_fn: Optional[Callable] = None,
        inference: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._message_transform = message_transform
        if not isinstance(tokenizer, Llama3Tokenizer):
            raise ValueError("tokenizer must be an instance of Llama3Tokenizer.")
        self._tokenizer = tokenizer

        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

        self._inference = inference

    @property
    def inference(self) -> bool:
        return self._inference

    @inference.setter
    def inference(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("inference must be a boolean.")
        self._inference = value

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        transformed_sample = self._message_transform(sample)
        if "messages" in transformed_sample:
            validate_messages(transformed_sample["messages"])

        # tokenize the messages and pack them into a dictionary of token and mask
        tokenized_dict = self._tokenizer(transformed_sample, inference=self._inference)

        if not ("tokens" in tokenized_dict and "mask" in tokenized_dict):
            keys_str = ", ".join(tokenized_dict.keys())
            error_message = (
                "tokenizer returned the following keys: " f"{keys_str}. Must return 'tokens' and 'mask' as keys."
            )
            raise ValueError(error_message)

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        tokenized_dict["labels"] = list(
            np.where(
                tokenized_dict["mask"],
                CROSS_ENTROPY_IGNORE_IDX,
                tokenized_dict["tokens"],
            )
        )
        assert len(tokenized_dict["tokens"]) == len(tokenized_dict["labels"])

        return tokenized_dict


class ASRInputOutputToMessages(Transform):
    """
    Message transform class that converts a single sample with "input" and "output" fields,
    (or equivalent fields specified in column_map) to user and assistant messages,
    respectively. This is useful for datasets that have two columns, one containing
    the user prompt string and the other containing the model response string::

        |  input          |  output          |
        |-----------------|------------------|
        | "user prompt"   | "model response" |

    Args:
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is False.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "input"
            and "output" column names to the actual column names in the dataset. Keys should
            be "input" and "output" and values should be the actual column names. Default is None,
            keeping the default "input" and "output" column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.

    Raises:
        ValueError: If ``column_map`` is provided and ``input`` not in ``column_map``, or
            ``output`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
    ):
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt
        if column_map:
            if "input" not in column_map:
                raise ValueError(f"Expected a key of 'input' in column_map but found {column_map.keys()}.")
            if "output" not in column_map:
                raise ValueError(f"Expected a key of 'output' in column_map but found {column_map.keys()}.")
            self._column_map = column_map
        else:
            self._column_map = {"input": "input", "output": "output"}

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        messages = [
            Message(
                role="user",
                content="".join(map(dsu2pua, sample[self._column_map["input"]])),
                masked=not self.train_on_input,
                eot=True,
            ),
            Message(
                role="assistant",
                content=sample[self._column_map["output"]],
                masked=False,
                eot=True,
            ),
        ]
        if self.new_system_prompt is not None:
            messages = [Message(role="system", content=self.new_system_prompt, masked=True, eot=True)] + messages
        return {"messages": messages}


def asr_instruct_dataset(
    tokenizer: Llama3Tokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    inference: bool = False,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Configure a custom dataset with user instruction prompts and model responses.

    This builder function can be used to configure a custom instruct dataset directly from the yaml config
    as an alternative to :class:`~torchtune.datasets.SFTDataset`, as it is made to be config friendly.

    The dataset should follow this format:

    .. code-block:: text

        |  input          |  output          |
        |-----------------|------------------|
        | "user prompt"   | "model response" |

    If your column names are different, you can use the ``column_map`` parameter to change
    the expected column names. For example, if your dataset has columns ``"question"`` and
    ``"answer"`` you can use::

        column_map = {"input": "question", "output": "answer"}

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``False`` by default
    - If ``train_on_input`` is True, the prompt is used during training and
    contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``, and set ``split="train"``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "input"
            and "output" column names to the actual column names in the dataset. Keys should be "input" and
            "output" and values should be the actual column names. Default is None, keeping the default "input"
            and "output" column names.
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is False.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.
        packed (bool): Whether or not to pack the dataset to tokenizer's ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.

    Examples:

    ::

        my_dataset.json
        [
            {
                "question": "What time is it in London?",
                "answer": "It is 10:00 AM in London.",
            },
            {
                ...
            },
            ...,
        ]

    ::

        >>> from torchtune.datasets import instruct_dataset
        >>> dataset = instruct_dataset(
        ...     tokenizer=tokenizer,
        ...     source="json",
        ...     data_files="my_dataset.json",
        ...     column_map={
        ...         "input": "question",
        ...         "output": "answer",
        ...     },
        ...     train_on_input=False,
        ...     packed=False,
        ...     split="train",
        ... )
        >>> tokens = dataset[0]["tokens"]
        >>> tokenizer.decode(tokens)
        "What time is it in London?It is 10:00 AM in London."

    This can also be accomplished via the yaml config:

    .. code-block:: yaml

        dataset:
          _component_: torchtune.datasets.instruct_dataset
          source: json
          data_files: my_dataset.json
          column_map:
            input: question
            output: answer
          train_on_input: False
          packed: False
          split: train

    Returns:
        Union[SFTDataset, PackedDataset]: the configured :class:`~torchtune.datasets.SFTDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``

    Raises:
        ValueError: If ``packed=True`` and ``tokenizer.max_seq_len`` is not set.
    """
    message_transform = ASRInputOutputToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        inference=inference,
        split=split,  # in effect one of the load_dataset_kwargs
        **load_dataset_kwargs,
    )

    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError("PackedDataset requires a max_seq_len to be set on the tokenizer.")
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds


# ASR Prompt Template for supervised finetuning and inference
ASR_SFT_PROMPT_DICT: dict[Role, tuple[str, str]] = {"user": ("English text: ", "\n---\nEnglish speech: ")}
ASR_SFT_PROMPT_TEMPLATE = PromptTemplate(ASR_SFT_PROMPT_DICT)
