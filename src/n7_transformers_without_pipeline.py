from transformers import AutoTokenizer, AutoModelForCausalLM  # Used to load the pre-trained model and tokenizer
from llm_mlops import track_time  

# Load a pre-trained causal language model (TinyLlama-1.1B-Chat-v1.0) from Hugging Face model hub
# The model is loaded and mapped to the GPU using 'cuda' for faster inference
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="cuda")

# Load the corresponding tokenizer for the model
# Tokenizer is used to convert the input text into numerical tokens that the model understands
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Define the chat message in a structured format
# The 'role' key indicates who is speaking (either the system or user)
# 'content' is the actual message text
messages = [
    {
        "role": "system",  # Defines the chatbot's role/behavior
        "content": "You are a friendly chatbot who is always helpful.",  # Sets the personality of the chatbot
    },
    {"role": "user", "content": "How can I fix my car's break?"},  # User input asking for advice on fixing car brakes
]

# Apply the chat template provided by the tokenizer to format the input messages for the model
# Parameters:
# - 'tokenize=False' means that the text will not be tokenized yet, but will retain the original text
# - 'add_special_tokens=False' ensures no additional tokens (like start/end) are added in this step
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)

# Tokenize the input prompt and convert it into tensors
# The tokenizer will split the text into tokens, then convert it into tensors that the model can process
# The prompt is duplicated 256 times to simulate batch processing, which helps in measuring throughput
# 'return_tensors="pt"' specifies that PyTorch tensors are returned, and '.to("cuda")' moves the tensors to the GPU
input_ids = tokenizer([prompt] * 256, return_tensors="pt").to("cuda")

# Measure the execution time using the 'track_time' utility
# Parameters passed here ('input_ids["input_ids"]') are the tokenized inputs for which the generation time will be tracked
with track_time(input_ids["input_ids"]):
    # Generate text using the model
    # The following parameters control the generation process:
    # - 'max_length=256': The maximum number of tokens to be generated in the output
    # - 'do_sample=True': Enables sampling to introduce randomness in token generation
    # - 'temperature=0.1': Low temperature means the model will generate more focused, less random responses
    # - 'top_k=50': Only the top 50 token candidates are considered for each step (helps in controlling the randomness)
    # - 'top_p=0.95': Nucleus sampling; considers the smallest group of tokens with a cumulative probability of 95%
    outputs = model.generate(**input_ids, max_length=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)

# Decode the generated output back into text using the tokenizer
# 'skip_special_tokens=True' removes any special tokens (like <EOS>, <PAD>) from the output to keep the text clean
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Performance stats:
# GPU (latency and throughput):
# Latency: 2.6578898429870605 seconds
# Throughput: 14.10 inputs/s

# Comment:
# Latency on GPU remains comparable to using the pipeline (about 2.6 seconds), but there is a significant improvement
# in throughput with this approach compared to the pipeline, processing 14.1 inputs per second.

"""
The script highlights the performance improvement of using a GPU for inference, 
particularly in terms of throughput, which is crucial for handling larger batch sizes or 
real-time applications. The throughput significantly increases when compared to running 
the model on a CPU or using standard pipelines, processing 14.1 inputs per second on the GPU.

"""