

# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer  # For loading the causal language model (autoregressive)
from datasets import load_dataset  # For loading datasets
import torch  # PyTorch for tensor and GPU operations
from tqdm.auto import tqdm  # For displaying a progress bar during iteration
from ctranslate2.converters import TransformersConverter  # Converts Hugging Face models to CTranslate2 format
from ctranslate2 import Generator  # Used for efficient model inference with CTranslate2

from contextlib import contextmanager  # For creating context managers (used to measure time)
import time 

# A utility function to track the execution time of a block of code
@contextmanager
def track_time():
     start = time.time()  # Record the starting time
     yield  # Allow execution of the block of code where this context manager is used
     end = time.time()  # Record the ending time
     print(f"Execution time: {end - start} seconds")  # Print the elapsed time in seconds


# Load the pre-trained Causal Language Model (a variant of GPT) and move it to GPU (cuda)
model = AutoModelForCausalLM.from_pretrained("TheFuzzyScientist/diabloGPT_open-instruct").to("cuda")

# Load the tokenizer, which handles converting text to token IDs and back to text
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")

# Set the padding token to be the same as the end-of-sequence (EOS) token
tokenizer.pad_token = tokenizer.eos_token

# Save the model and tokenizer locally so they can be converted to CTranslate2 format
model.save_pretrained("models/gpt-instruct")  # Save the model to the directory
tokenizer.save_pretrained("models/gpt-instruct")  # Save the tokenizer

# Convert the model to CTranslate2 format for efficient inference
# The "quantization" argument reduces model precision to 16-bit floating point (float16) for faster computation
converter = TransformersConverter("models/gpt-instruct")
out_path = converter.convert(output_dir="models/gpt-instruct-quant", quantization="float16")

# Load the quantized model with CTranslate2 for inference
generator = Generator("models/gpt-instruct-quant", device="cuda")  # Use the quantized model on the GPU (cuda)

# Load the dataset, which consists of instructions used for language model training
dataset = load_dataset("hakurei/open-instruct-v1", split="train")
dataset = dataset.to_pandas()  # Convert the dataset to a Pandas DataFrame for easier manipulation

# Sample 3000 random prompts (instructions) from the dataset
prompts = dataset["instruction"].sample(3000, random_state=42).tolist()

# ------------------- Normal Batching (with full-precision model) -------------------

# Function to split the input data into chunks of a specified size (batching)
def chunker(seq, size):
     return (seq[pos : pos + size] for pos in range(0, len(seq), size))

# Function to generate tokens (output) from the input prompts using the full-precision model
def batch_generate_tokens(tokens):
     # Generate tokens from the model with constraints:
     # - max_length: the maximum length of generated text (256 tokens)
     # - num_beams: beam search with 2 beams (better accuracy)
     # - repetition_penalty: penalizes repeating tokens (discourages repeating the same words)
     outputs = model.generate(tokens, max_length=256, pad_token_id=tokenizer.eos_token_id, num_beams=2, repetition_penalty=1.5)

     # Decode the generated tokens back into human-readable text, skipping special tokens (e.g., EOS)
     return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Function to perform prediction on batches of prompts
def predict_batch(prompts, batch_size):
     # Tokenize the input prompts, return them as PyTorch tensors, pad/truncate to 128 tokens, and batch them
     inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)["input_ids"]

     # Process the tokenized input in chunks/batches of the specified size
     for batch in chunker(inputs, batch_size):
          yield batch_generate_tokens(batch.to(model.device))  # Move the batch to the GPU for inference

# Measure the execution time for normal batching (full-precision model inference)
with track_time():
     for batch_prediction in tqdm(predict_batch(prompts, 32)):  # Predict with batch size of 32
          continue  # We're not interested in printing the output here, just measuring the time

# Execution time: 242.11 seconds (normal model inference)

# ------------------- CTranslate2 Batching (with quantized model) -------------------

# Function to perform prediction using the CTranslate2 quantized model (faster inference)
def batch_generate_ctrans(prompts, batch_size):
     # Tokenize each prompt, truncating the input to a maximum length of 128 tokens
     inputs = [tokenizer.tokenize(prompt, truncation=True, max_length=128) for prompt in prompts]

     # Generate tokens using the quantized model, with constraints:
     # - max_length: the maximum length of the generated text (256 tokens)
     # - max_batch_size: maximum number of prompts processed at once (32)
     # - beam_size: beam search with 2 beams for accuracy
     # - repetition_penalty: penalizes repeating tokens (prevents repeated phrases)
     results = generator.generate_batch(inputs, max_length=256, max_batch_size=batch_size, beam_size=2, repetition_penalty=1.5)

     # Extract the generated token IDs from the results and decode them back into text
     result_ids = [res.sequences_ids[0] for res in results]  # Get the generated sequences from the results
     return tokenizer.batch_decode(result_ids, skip_special_tokens=True)  # Decode token IDs back to readable text

# Cleanup: Delete the full-precision model to free GPU memory, and clear the GPU cache
del model
torch.cuda.empty_cache()

# Measure the execution time for CTranslate2 batching (quantized model inference)
with track_time():
     batch_generate_ctrans(prompts, 32)  # Predict with batch size of 32

# Execution time: 150.97 seconds (faster inference with quantized model)

"""
Detailed Explanation:
Quantization for Model Efficiency:
CTranslate2 is used for model quantization, which reduces the precision of the model's weights 
(in this case, from 32-bit to 16-bit float precision using float16).
Quantization is a key technique to make the model more efficient by reducing memory usage and 
speeding up computations, especially on GPUs or specialized hardware.
By reducing the size of the weights and using efficient data structures, CTranslate2 
optimizes the inference process. This allows the model to be used more effectively during inference 
without sacrificing too much accuracy.

Key Components:
Model Loading and Conversion:

The Hugging Face AutoModelForCausalLM is loaded and then converted to CTranslate2 format using the 
TransformersConverter.
The model is quantized to float16 precision, reducing its memory footprint and increasing inference speed.
Normal Batching (with full-precision model):

Prompts are tokenized and processed in batches of 32, with each batch being sent to 
the model for inference.
This process takes significantly longer because the model is using the full-precision weights, 
which require more memory and computation time.
CTranslate2 Batching (with quantized model):

In contrast to normal batching, here the model has been quantized. This allows faster token generation, 
even with the same batch size and beam search parameters.
The use of quantized weights and optimized inference methods provided by CTranslate2 results in a 
notable reduction in execution time (~150.97 seconds compared to ~242.11 seconds).
By using CTranslate2 and quantization, the code demonstrates a significant performance boost, 
highlighting the benefits of optimizing model inference, especially for large-scale natural 
language generation tasks.

"""