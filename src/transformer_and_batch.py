"""
Note:
some parts of your code are related to GPT-2 architecture, specifically when using 
causal language models (CLMs) - which are both the model and tokenizer:

model = AutoModelForCausalLM.from_pretrained("TheFuzzyScientist/diabloGPT_open-instruct").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")

the tokenizer and model in your code (specifically the "DialoGPT-medium" tokenizer and the 
AutoModelForCausalLM setup) are fine-tuned versions of GPT-2, designed for generating coherent 
text sequences based on previous inputs.

"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm.auto import tqdm

# Load pre-trained causal language model and move it to GPU (CUDA)
model = AutoModelForCausalLM.from_pretrained("TheFuzzyScientist/diabloGPT_open-instruct").to("cuda")

# Load pre-trained tokenizer and set padding to the left
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to the EOS token

# Load dataset from Hugging Face and convert it to a pandas DataFrame
dataset = load_dataset("hakurei/open-instruct-v1", split="train")
dataset = dataset.to_pandas()

# Function to generate text from a single prompt
def generate_text(prompt):
     inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)  # Tokenize input and move it to GPU
     outputs = model.generate(inputs, max_length=64)  # Generate text with a max length of 64 tokens
     generated = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode output to readable text

     return generated[: generated.find(".") + 1]  # Return text up to the first period

# Example usage of the generate_text function
generate_text("What's the best way to cook chicken breast?")

# Function to batch-generate texts for multiple prompts at once
def batch_generate_texts(prompts):
     inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)["input_ids"]  # Batch tokenize and pad input
     outputs = model.generate(inputs, max_length=64, pad_token_id=tokenizer.eos_token_id)  # Generate text for the batch
     generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)  # Decode all generated outputs

     return generated

# Testing batch generation with different batch sizes
batch_generate_texts(dataset["instruction"][:1].tolist())  # Generate for the first prompt
batch_generate_texts(dataset["instruction"][:20].tolist())  # Generate for the first 20 prompts
batch_generate_texts(dataset["instruction"][:100].tolist())  # Generate for the first 100 prompts
batch_generate_texts(dataset["instruction"][:200].tolist())  # Generate for the first 200 prompts
# batch_generate_texts(dataset["instruction"].sample(200).tolist()) # This might crash due to memory overload

# Dynamic batching function to handle generation based on token size limits
def batch_generate_tokens(tokens):
     outputs = model.generate(torch.stack(tokens), max_length=64, pad_token_id=tokenizer.eos_token_id)  # Stack and generate

     return tokenizer.batch_decode(outputs, skip_special_tokens=True)  # Decode the generated outputs

# Function to handle dynamic batching with a max token size limit
def dynamic_batching(prompts, max_tokens, is_pretokenized=False):
     if not is_pretokenized:
          # Tokenize the input and add padding
          tokenized_texts = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"].to(model.device)
     else:
          # If pre-tokenized, just use the provided tokens
          tokenized_texts = prompts

     current_batch = []  # Initialize current batch
     current_batch_size = 0  # Track the size of the current batch

     # Loop through tokenized text to dynamically batch based on max token size
     for tokenized_text in tokenized_texts:
          if current_batch_size + len(tokenized_text) > max_tokens and current_batch:
               yield batch_generate_tokens(current_batch)  # Generate tokens for the current batch

               # Reset the batch for the next round
               current_batch, current_batch_size = [], 0

          current_batch.append(tokenized_text)  # Add the current tokenized text to the batch
          current_batch_size += len(tokenized_text)  # Update the current batch size

     # Process the final batch if any tokens are left
     if current_batch:
          yield batch_generate_tokens(current_batch)

# Example generator for dynamically batching text generation
generator = dynamic_batching(dataset["instruction"][:40].tolist() * 1000, 3200)

from contextlib import contextmanager
import time


# =============
# Batch Sorting and tracking the time
# Context manager to track execution time of code blocks
@contextmanager
def track_time():
     start = time.time()  # Record start time
     yield
     end = time.time()  # Record end time
     print(f"Execution time: {end - start} seconds")  # Print the time taken

# Run the dynamic batching generator and track time
with track_time():
     for batch_predictions in tqdm(generator):
          continue  # Continue to the next batch

# Function to sort prompts by token length and batch them dynamically
def sort_batches(prompts, max_tokens):
     tokenized_texts = tokenizer(prompts, padding=False)["input_ids"]  # Tokenize without padding
     sorted_tokens = sorted(tokenized_texts, key=len)  # Sort tokens by their length

     sorted_batches = {}  # Dictionary to store sorted batches
     for sorted_token in sorted_tokens:
          length = len(sorted_token)
          if length not in sorted_batches:
               sorted_batches[length] = []  # Initialize batch for each length category

          sorted_batches[length].append(sorted_token)  # Add tokenized text to the correct batch

     # Generate predictions for each batch of similar-length tokens
     for length, sorted_batch in sorted_batches.items():
          tensor_batch = torch.stack([torch.tensor(sorted_token) for sorted_token in sorted_batch]).to(model.device)
          for batch_prediction in dynamic_batching(tensor_batch, max_tokens=max_tokens, is_pretokenized=True):
               yield batch_prediction  # Yield batch predictions

# Example usage of the sort_batches function
generator = sort_batches(dataset["instruction"][:40].tolist() * 1000, 3200)

# Track time and print the number of predictions for each batch
with track_time():
     for batch_predictions in tqdm(generator):
          print(len(batch_predictions))  # Print the size of each batch
