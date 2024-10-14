# Importing RabbitBuffer from the 'n1_rabbit_mq' module to handle communication with RabbitMQ.
# This will be used to queue messages for the language model in a distributed setup.
from llm_mlops.n1_rabbit_mq import RabbitBuffer

# Importing AutoTokenizer from the Hugging Face Transformers library.
# This tokenizer is responsible for converting text input into tokens that can be processed by the language model.
from transformers import AutoTokenizer

# Defining a list of messages in the format used for chat-like applications.
# The 'system' message sets the tone for the chatbot, making it a helpful assistant.
# The 'user' message contains the actual query from the user.
messages = [
     {
          "role": "system",
          "content": "You are a friendly chatbot who is always helpful.",  # Instruction to define chatbot's behavior
     },
     {"role": "user", "content": "How can I get rid of a llama on my lawn?"},  # The user's query or prompt
]

# Initializing the tokenizer using a pre-trained model ("TinyLlama-1.1B-Chat-v1.0") from local model files.
# The tokenizer will convert the chat messages into input tokens for the model.
tokenizer = AutoTokenizer.from_pretrained("models/TinyLlama-1.1B-Chat-v1.0")

# Applying a chat template to format the messages for the model.
# This merges the 'system' and 'user' messages into a single text string in a format the model understands.
# tokenize=False: Leaves the text as plain text without tokenizing it yet.
# add_special_tokens=False: Doesn't add any special tokens like <sos> (start-of-sequence) or <eos> (end-of-sequence).
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)

# Instantiating the RabbitBuffer class to create a queue named "llama-queue".
# RabbitBuffer will be used to produce (send) and consume (retrieve) messages from this queue using RabbitMQ.
buffer = RabbitBuffer("llama-queue")

# Using the produce method to send 100,000 copies of the generated prompt to the RabbitMQ queue.
# This simulates bulk message sending, where multiple instances of the same prompt are being queued for processing.
buffer.produce([prompt] * 100_000)

# Using the consume method to retrieve 10 messages from the queue.
# This simulates consuming a small batch of messages from the queue.
buffer.consume(10)

"""
How This Works in a Distributed Environment:
Message Queueing with RabbitMQ: The RabbitBuffer class uses RabbitMQ to queue and distribute messages. 
This is useful in a distributed environment because:
It decouples the message producer (the code that sends chat prompts) from the message 
consumer (the language model that processes the prompts).
Allows multiple instances of the model to process messages in parallel, improving throughput 
and performance.
"""