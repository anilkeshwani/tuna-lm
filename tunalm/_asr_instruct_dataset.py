# based on the torchtune library:
# from torchtune.data import InputOutputToMessages

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Union

from sardalign.utils import dsu2pua
from torchtune.data import Message
from torchtune.data._utils import format_content_with_images, load_image
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


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
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
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
        model_transform=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )

    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError("PackedDataset requires a max_seq_len to be set on the tokenizer.")
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
