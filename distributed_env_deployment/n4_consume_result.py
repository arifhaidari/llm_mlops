# Import the `RabbitBuffer` class from the `llm_mlops.n1_rabbit_mq` module.
# The `RabbitBuffer` class is used to interact with a RabbitMQ queue for message passing in distributed systems.
# In this case, it handles retrieving (consuming) messages from the queue.

from llm_mlops.n1_rabbit_mq import RabbitBuffer

# Initialize a new `RabbitBuffer` instance for consuming results from the "llama-results" queue.
# The "llama-results" queue contains the generated outputs of a language model processed in a previous task.
buffer = RabbitBuffer("llama-results")

# Consume (retrieve) 10,000 messages from the "llama-results" queue.
# The `consume(10_000)` method retrieves up to 10,000 messages at once from the queue.
# Each message represents a generated result from the model, stored in binary format, which needs decoding.
# We could consume the entire queue or specify a smaller number of messages (e.g., 20) if we want to retrieve only a few.
results = buffer.consume(10_000) 

# Optionally, we could use a smaller number of messages by uncommenting the following line.
# For example, consume only 20 messages from the queue:
# results = buffer.consume(20)

# Check the total number of messages retrieved from the queue by printing the length of the `results` list.
len(results)

# Print the 9000th result in the list of retrieved messages.
# Each message is stored in binary format, so we use `.decode()` to convert it into a human-readable string.
print(results[9000].decode())
