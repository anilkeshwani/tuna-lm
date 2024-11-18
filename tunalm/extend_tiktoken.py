import base64
import logging
import os
from pathlib import Path

from sardalign.utils import dsu2pua


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def add_dsus_to_tiktoken(tokenizer_model: Path, n_new_dsus: int):
    """
    Appends new base64-encoded tokens to a base64-encoded ASCII tokenizer.model file as used by tiktoken.

    Arguments:
        - tokenizer_model: Path to the tokenizer.model file
        - n_new_dsus: Number of DSUs to add as tokens. Converted to PUA tokens via dsu2pua.
    """
    with open(tokenizer_model, "r") as file:
        lines = file.readlines()

    # Obtain the new DSUs to add
    new_dsu_tkns = [dsu2pua(i) for i in range(n_new_dsus)]  # NOTE in future can specify start/end idxs for new DSUs

    # Create a dict[bytes, int] dictionary of the current vocabulary - to test for duplicates
    vocabulary: dict[bytes, int] = {}
    for line in lines:
        token, rank = line.split()
        vocabulary[base64.b64decode(token.encode("utf-8"))] = int(rank)

    # Get next merge rank
    rank = max(vocabulary.values()) + 1  # in case tokenizer.model is not sorted by merge rank

    # Prepare new lines with base64-encoded tokens
    new_tokenizer_lines = []
    for i, token in enumerate(new_dsu_tkns):
        token_bytes: bytes = token.encode("utf-8")
        if token_bytes in vocabulary:
            logger.warning(f"Token {token} (idx: {i}) already exists in the vocabulary")
            continue
        token_b64_ascii = base64.b64encode(token_bytes).decode("utf-8")
        new_tokenizer_lines.append(f"{token_b64_ascii} {rank}\n")
        rank += 1

    # Append the new lines to the file
    with open(tokenizer_model, "a") as file:
        file.writelines(new_tokenizer_lines)

    print(f"Added {len(new_tokenizer_lines)} tokens to {tokenizer_model}")


if __name__ == "__main__":
    # Usage Example
    PRETRAINED_MODELS_DIR = Path(os.environ.get("HAFH", "/mnt/scratch-artemis/anilkeshwani/")) / "models/"
    tokenizer_model = PRETRAINED_MODELS_DIR / "base-torchtune/_Llama-3.2-1B/original/tokenizer.model"
    n_new_dsus = 5000  # Replace with the number of new DSUs to add
    add_dsus_to_tiktoken(tokenizer_model, n_new_dsus)
