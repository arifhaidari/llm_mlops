from transformers import pipeline 
from llm_mlops.utils import track_time 

# Create a text generation pipeline using the "TinyLlama-1.1B-Chat-v1.0" model and specify "cuda" for GPU usage
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="cuda")

# Define chat messages. The format follows a typical role-based chat where:
# - "system" sets the behavior of the chatbot
# - "user" is the input from the user
messages = [
     {
          "role": "system",
          "content": "You are a friendly chatbot who is always helpful.",  # Chatbot's role defined by the system
     },
     {"role": "user", "content": "How can I fix my car's break?"},  # User asking a question to the chatbot
]

# Format the chat messages using the tokenizer's chat template to prepare them for generation.
# The template is applied without tokenizing the text, and a generation prompt is added for the model to respond.
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Create a list of the same prompt to simulate multiple requests (used to measure throughput)
prompts = [prompt] * 5  # The prompt is repeated 5 times

# Measure the time taken to generate responses using the track_time utility (to compare latency between CPU and GPU)
with track_time(prompts):
     # Generate the responses from the model. The generation parameters are as follows:
     # - max_new_tokens: Maximum number of new tokens to generate (256 tokens in this case)
     # - do_sample: Enables random sampling (required for diverse text generation)
     # - temperature: Controls randomness in sampling (lower values make the model more conservative)
     # - top_k and top_p: Techniques to limit the sampling space to improve response diversity and relevance
     outputs = pipe(prompts, max_new_tokens=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)

# Print the first generated response (the output is structured as a list of lists)
print(outputs[0][0]["generated_text"])

## Execution results (comparison between CPU and GPU):
## CPU latency: 48 seconds
## GPU latency: 2.907 seconds
## GPU throughput: 0.34 inputs per second

"""
The GPU drastically reduces the latency (2.9 seconds on GPU vs. 48 seconds on CPU), 
showcasing the advantage of using distributed computing with transformers on GPUs for 
faster inference and better throughput.
"""

print('end of the code')
