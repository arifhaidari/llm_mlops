from llm_mlops.utils import track_time
from transformers import AutoTokenizer
from ctranslate2 import Generator

"""
Generator is a class from the CTranslate2 library, which is optimized for fast and efficient 
inference of neural networks, especially transformer models like GPT and BERT. 
It converts models to a more efficient format, improving both latency (how fast a single 
inference completes) and throughput (how many inferences can be processed per second).
"""

# Load the pre-trained tokenizer for the TinyLlama model, which specializes in chatbot conversations.
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load the model using CTranslate2's Generator, specifying the pre-converted TinyLlama model located in the 'models' directory.
# 'device="cuda"' ensures that the model runs on the GPU, improving performance for inference tasks.
model = Generator("models/TinyLlama-1.1B-Chat-v1.0-ctrans", device="cuda")

# Define the chatbot interaction with a system and user role.
messages = [
    {
        "role": "system",  
        "content": "You are a friendly chatbot who is always helpful.",  
    },
    {"role": "user", "content": "How can I get rid of a llama on my lawn?"},
]

# Format the conversation as a prompt using the tokenizer's chat template.
# 'tokenize=False' means the tokenizer does not yet break down the prompt into tokens at this stage.
# 'add_special_tokens=False' prevents special tokens (e.g., start/end tokens) from being added automatically.
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)

# Tokenize the prompt using the tokenizer. This converts the text into token IDs (numerical representations).
# The prompt is repeated 256 times to simulate batch inference, which helps measure throughput.
input_tokens = [tokenizer.tokenize(prompt)] * 256

# Track the time it takes to generate a batch of outputs using CTranslate2's 'generate_batch' function.
# 'input_tokens' are passed to the model, which generates predictions based on them.
with track_time(input_tokens):
    outputs = model.generate_batch(input_tokens)


# Extract the generated token IDs from the model's output.
# 'sequences_ids' contains the tokenized sequences generated by the model.
results_ids = [res.sequences_ids[0] for res in outputs]

# Decode the generated tokens back into human-readable text using the tokenizer.
# 'skip_special_tokens=True' ensures that special tokens (like EOS or PAD) are removed from the decoded output.
outputs = tokenizer.batch_decode(results_ids, skip_special_tokens=True)

# Print the 100th generated response from the batch.
print(outputs[100])

"""
results_ids: Extracts the generated token sequences from the model output. Each sequence represents the model's response to the prompt.
batch_decode: Converts the generated token IDs back into human-readable text, skipping special tokens.
"""

# Performance Measurements:
# with gpu
# latency: 1.3013768196105957s
# throughput: 31.77 inputs/s

# using ctranslate2, both latency and throughput improved.

"""
What is CTranslate2?
CTranslate2 is an optimized inference engine designed to reduce the computational load and 
improve the speed of transformer-based models. It specifically targets efficiency in 
model execution, both for CPU and GPU. CTranslate2 achieves this through:

Quantization: Reducing the precision of model weights (e.g., from FP32 to FP16 or even INT8), 
which decreases memory usage and speeds up computation without significantly impacting accuracy.
Batching optimizations: Efficiently handles multiple inputs at once, improving throughput 
when processing larger batches.
GPU/CPU optimization: Designed to leverage hardware acceleration (especially GPUs) more efficiently 
than traditional models.

Why CTranslate2 Improved Latency and Throughput:
Quantization: The model uses lower-precision weights (FP16 or INT8), which require less memory 
and are faster to compute.
Efficient GPU Utilization: By running on the GPU (device="cuda"), the model can process multiple 
inputs in parallel, leading to lower latency and higher throughput.
Optimized Batching: CTranslate2 generate_batch function is highly efficient for processing 
multiple inputs simultaneously, which results in significant throughput gains (31.77 inputs/second).
"""