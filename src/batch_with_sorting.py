
"""
The code first performs text generation in batches without sorting, and then compares the 
performance when prompts are sorted by length before batching. The execution times show the 
efficiency gained from sorted batching.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm.auto import tqdm  # tqdm for showing progress bars
from contextlib import contextmanager
import time  # used for tracking execution time

# A helper function to track the time it takes to run a block of code
@contextmanager
def track_time():
    start = time.time()  # Start the timer
    yield  # Allows execution of the block of code where the context manager is used
    end = time.time()  # End the timer
    print(f"Execution time: {end - start:.2f}s")  # Print the elapsed time in seconds

# Load the pre-trained autoregressive model (Causal Language Model) into GPU (cuda)
model = AutoModelForCausalLM.from_pretrained("TheFuzzyScientist/diabloGPT_open-instruct").to("cuda")

# Load the tokenizer, which converts text to token IDs and vice versa, setting padding to the left
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")

# Set the padding token to be the same as the EOS token (end of sequence) to ensure consistency
tokenizer.pad_token = tokenizer.eos_token

# Load a dataset from Hugging Face and convert it to a Pandas DataFrame for easy sampling
dataset = load_dataset("hakurei/open-instruct-v1", split="train")
dataset = dataset.to_pandas()

# Sample 4 random prompt examples from the dataset
prompts = dataset["instruction"].sample(4).tolist()

# Tokenize the prompts, padding them to have equal length, and return their token IDs
inputs = tokenizer(prompts, padding=True)["input_ids"]

# Decode the tokenized inputs back to text for printing (replace EOS token with "[PAD]")
print("\n\n".join(tokenizer.batch_decode(inputs)).replace(tokenizer.eos_token, "[PAD]"))

# ----------- Normal Batching (no sorting) ---------------

# Function to break a sequence into smaller chunks/batches of a fixed size
def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))

# Function to generate text from the tokenized input
def batch_generate_tokens(tokens):
    # The model generates tokens given input tokens, with a maximum of 64 new tokens per input
    outputs = model.generate(tokens, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated tokens back to human-readable text and skip any special tokens
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Function to process prompts in batches of a specified size
def predict_batch(prompts, batch_size):
    # Tokenize all prompts, padding them to a maximum length of 512 tokens
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"]

    # Split the tokenized input into smaller batches and generate text for each batch
    for batch in chunker(inputs, batch_size):
        yield batch_generate_tokens(batch.to(model.device))  # Move batch to GPU and generate tokens

# Sample 3000 random prompts from the dataset for prediction
prompts = dataset["instruction"].sample(3000).tolist()

# Time the execution for batch prediction without sorting
with track_time():
    # Iterate over the batch predictions and print the number of generated responses in each batch
    for batch_prediction in tqdm(predict_batch(prompts, 32)):
        print(len(batch_prediction))

# Execution time: Longer as with sorted (without sorted batching)

# ----------- Sorted Batching (optimized for efficiency) ---------------

# Function to process prompts with sorted batching, where inputs are sorted by length
def predict_sorted_batches(prompts, max_batch_size):
    # Tokenize the prompts without padding and truncate to a maximum length of 512 tokens
    inputs = tokenizer(prompts, padding=False, truncation=True, max_length=512)["input_ids"]

    # Sort the tokenized inputs by their length (shorter inputs come first)
    sorted_tokens = sorted(inputs, key=len)
    
    # Create a dictionary to group sorted tokenized inputs by their length
    sorted_batches = {}
    for sorted_input in sorted_tokens:
        if not len(sorted_input):  # Skip empty sequences
            continue

        length = len(sorted_input)
        if length not in sorted_batches:
            sorted_batches[length] = []  # Initialize an empty list for each unique length

        sorted_batches[length].append(sorted_input)  # Append tokenized input to the appropriate length group

    # For each unique length, process the batches
    for length, sorted_batch in sorted_batches.items():
        # Chunk the sorted batches into smaller chunks of max_batch_size and generate text for each chunk
        for batch in chunker(sorted_batch, max_batch_size):
            tensor_batch = torch.tensor(batch).to(model.device)  # Convert to a tensor and move to GPU
            yield batch_generate_tokens(tensor_batch)

# Time the execution for batch prediction with sorted batching
with track_time():
     # Iterate over the sorted batch predictions and print the number of generated responses in each batch
     for batch_prediction in tqdm(predict_sorted_batches(prompts, 32)):
          print(len(batch_prediction))

# Execution time: half of Normal batching (with sorted batching)
