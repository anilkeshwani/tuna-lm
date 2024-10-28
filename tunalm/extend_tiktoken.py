import base64
from pprint import pp

import tiktoken


enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")

print(f"{type(enc) = }")
# print(f"{len(enc) = }") # TypeError: object of type 'Encoding' has no len()

# print(f"{dir(enc) = }")

# print(f"{enc._mergeable_ranks = }")


# with open("/Users/anilkeshwani/Desktop/models/llamas/checkpoints/Llama3.2-3B/tokenizer.model", "rb") as f:
#     base64.b64encode(f.read()).decode("utf-8")


sample_base64 = ["IQ==", "Ig==", "Iw==", "JA==", "JQ=="]
decoded_tokens = [base64.b64decode(token).decode("utf-8") for token in sample_base64]
print(decoded_tokens)


def add_tokens_to_tokenizer(file_path, new_tokens):
    """
    Appends new base64-encoded tokens to a tokenizer.model file.

    Parameters:
    - file_path: Path to the tokenizer.model file
    - new_tokens: List of string tokens to add
    """
    # Read the current file to determine the highest ID
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the current highest ID
    last_line = lines[-1]
    highest_id = int(last_line.split()[1])
    next_id = highest_id + 1

    # Prepare new lines with base64-encoded tokens
    new_lines = []
    for token in new_tokens:
        encoded_token = base64.b64encode(token.encode("utf-8")).decode("utf-8")
        new_lines.append(f"{encoded_token} {next_id}\n")
        next_id += 1

    # Append the new lines to the file
    with open(file_path, "a") as file:
        file.writelines(new_lines)

    print(f"Added {len(new_tokens)} tokens to {file_path}")


# Usage Example
file_path = "tokenizer.model"
new_tokens = ["new_token_1", "new_token_2", "new_token_3"]  # Replace with your new tokens
add_tokens_to_tokenizer(file_path, new_tokens)
