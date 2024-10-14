from llm_mlops.utils import track_time

# Importing the LLM (Large Language Model) class and SamplingParams class from vllm.
# LLM is used for interacting with a transformer-based language model, and SamplingParams is used to control text generation parameters.
from vllm import LLM, SamplingParams

# Initialize the language model by loading the pre-trained "TinyLlama-1.1B-Chat-v1.0" model from the specified path.
# vLLM is an optimized language model inference library which accelerates the process of text generation.
llm = LLM(model="models/TinyLlama-1.1B-Chat-v1.0")

# Retrieve the tokenizer associated with the loaded language model.
# The tokenizer converts text into tokenized representations that the model can process and vice versa.
tokenizer = llm.get_tokenizer()

# Define a structured conversation. This structure is often used in chatbot interactions.
# The system message defines the behavior of the chatbot, while the user message contains the question for the chatbot.
messages = [
    {
        "role": "system",  # The system's message defines the chatbot's behavior.
        "content": "You are a friendly chatbot who is always helpful.",
    },
    {
        "role": "user",  # The user's message provides the input to the chatbot.
        "content": "How can I fix my car's break?",  # This is the user's question.
    },
]

# Apply the chat template from the tokenizer to format the conversation in the format that the model expects.
# tokenize=False ensures that the returned prompt is still in raw text form.
# add_special_tokens=False prevents any additional tokens (like start or end of sequence tokens) from being added.
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)

# Define the sampling parameters for text generation:
# max_tokens=256: The maximum number of tokens the model can generate in the output.
# seed=42: A fixed seed to make results reproducible (ensures that the same prompt produces the same results each time).
# temperature=0: A lower temperature makes the model more deterministic by reducing the randomness in its outputs.
# Note: vLLM likes the parameter to be separated.
sampling = SamplingParams(max_tokens=256, seed=42, temperature=0)

# Create a batch of prompts by repeating the same formatted prompt 1024 times.
# This simulates batch inference to test the model's throughput (how many prompts it can process at once).
prompts = [prompt] * 1024

# Measure the time (latency) it takes to generate text for all 1024 prompts.
# track_time records the time taken by the generate method.
with track_time(prompts):
    outputs = llm.generate(prompts, sampling)

# Extract the text output from the first generated sequence in the output for each prompt.
# Each result corresponds to the chatbot's response to the input prompt.
results = [output.outputs[0].text for output in outputs]

# Print the 1000th result from the batch, showing one of the chatbot's responses.
print(results[1000])

# Performance metrics:
# Latency: 0.704 seconds, which is the time it takes to generate responses for the batch.
# Throughput: 68.22 inputs per second, which indicates the number of prompts processed per second.

# Overall, using the vLLM library improved both latency and throughput by almost twice compared to the ctranslate2 library.

"""
Imported Packages and Their Role in Improving Performance:
vLLM (from vllm):

vLLM is a high-performance inference library designed specifically for large language models. 
It optimizes the execution of transformer models by:
Memory management optimization: vLLM uses techniques such as continuous batching and lazy weight 
loading to efficiently handle large batches of inputs.
Efficient GPU utilization: vLLM is highly optimized for GPU usage, 
which significantly improves both latency and throughput.
Latency: vLLM minimizes the time required to generate responses by leveraging fast 
token generation techniques.
Throughput: vLLM can handle larger batches of prompts more efficiently, leading to a 
throughput of 68.22 inputs/second, which is a substantial improvement over other 
frameworks like ctranslate2.
SamplingParams (from vllm):

Defines the parameters that control the generation of text (e.g., token count, randomness). 
This allows precise control over how text is generated, impacting both performance and output quality.
track_time (from llm_mlops.utils):

This utility function measures the execution time (latency) of the model. 
It is helpful in performance evaluation and benchmarking when comparing different models or configurations.
How vLLM Improved Latency and Throughput:
Optimized Memory Management: vLLM reduces the amount of memory required by efficiently 
loading and offloading model weights during inference.
Efficient Batching: It processes larger batches of inputs simultaneously, 
resulting in significantly higher throughput (68.22 inputs/second).
GPU Acceleration: By using GPU-based execution, vLLM minimizes latency, 
achieving a 0.704-second inference time for 1024 inputs.
Parallel Execution: vLLM utilizes multi-threaded execution and parallelism to 
distribute the work efficiently across available GPU resources.
This combination of optimizations in vLLM allows it to outperform other frameworks 
like ctranslate2 by almost doubling throughput and improving latency.
"""
