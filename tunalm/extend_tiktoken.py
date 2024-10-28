import base64
import warnings
from pathlib import Path

from sardalign.utils import dsu2pua


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
    new_lines = []
    for token in new_dsu_tkns:
        token_b64_bytes: bytes = base64.b64encode(token.encode("utf-8"))
        if token_b64_bytes in vocabulary:
            warnings.warn(f"Token {token} already in vocabulary")
            continue
        token_b64_ascii = token_b64_bytes.decode("utf-8")
        new_lines.append(f"{token_b64_ascii} {rank}\n")
        rank += 1

    # Append the new lines to the file
    with open(tokenizer_model, "a") as file:
        file.writelines(new_lines)

    print(f"Added {len(new_dsu_tkns)} tokens to {tokenizer_model}")


# Usage Example
tokenizer_model = "/mnt/scratch-artemis/anilkeshwani/models/base-torchtune/Llama-3.2-3B/original/tokenizer.model"
tokenizer_model = Path(tokenizer_model)
n_new_dsus = 200  # Replace with the number of new DSUs to add
add_dsus_to_tiktoken(tokenizer_model, n_new_dsus)
