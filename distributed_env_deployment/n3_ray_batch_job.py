# `ray` is a library for distributed computing and parallel processing.
# We use `RabbitBuffer` from the `llm_mlops.n1_rabbit_mq` module to handle RabbitMQ messaging for distributed tasks.
# `LLM` from `vllm` is used to load the language model for generating outputs from input prompts.
# `SamplingParams` sets parameters for sampling text generation (like max tokens, temperature, etc.).

# Install Ray if it's not installed:
# pip install -U "ray[default]"
import ray
from llm_mlops.n1_rabbit_mq import RabbitBuffer

# Initialize Ray in cluster mode with an auto-discovered address.
# This enables the use of Ray's distributed execution framework, allowing the program to run on multiple machines.
ray.init(address="auto")

from vllm import LLM, SamplingParams

# Define a remote function using the `@ray.remote` decorator.
# This function can be executed in parallel across distributed nodes (machines or GPUs) managed by Ray.
@ray.remote
def predict_batch():
    # Initialize RabbitBuffer to interact with RabbitMQ and consume messages from the "llama-queue".
    buffer = RabbitBuffer("llama-queue")

    # Consume 5000 messages (text prompts) from the "llama-queue".
    # Messages are stored in binary format, so we decode them into readable text.
    messages = buffer.consume(5000)
    prompts = [m.decode() for m in messages]

    # Set the sampling parameters for text generation.
    # max_tokens: Limits the generated text to 256 tokens.
    # seed: Sets a random seed to ensure reproducibility.
    # temperature: Controls randomness in generation. Lower values make the model more deterministic.
    sampling = SamplingParams(max_tokens=256, seed=42, temperature=0)

    # Load the language model (LLM) from a specific directory.
    # This model is used to generate the chatbot responses.
    llm = LLM(model="/root/ml-deployment/models/TinyLlama-1.1B-Chat-v1.0")

    # Generate responses based on the consumed prompts using the language model.
    outputs = llm.generate(prompts, sampling)

    # Extract the generated text from the model's output.
    # Only the first output in each sequence is selected (`output.outputs[0].text`).
    results = [output.outputs[0].text for output in outputs]

    # Initialize another RabbitBuffer to produce (send) the generated results to a different queue named "llama-results".
    result_buffer = RabbitBuffer("llama-results")
    result_buffer.produce(results)

    # Return the results, which will be collected by the Ray job.
    return results

# Main execution block.
if __name__ == "__main__":
    # Submit the `predict_batch` function to Ray for distributed execution.
    # `options` specifies that this job requires 1 GPU and 1 CPU to run.
    future = predict_batch.options(num_gpus=1, num_cpus=1).remote()

    # Wait for the job to finish and retrieve the results.
    # `ray.get()` blocks execution until the remote function completes and returns its output.
    ray.get(future)

    # Shut down the Ray cluster after the job is complete to free up resources.
    ray.shutdown()

# Command to submit this job using Ray's job submission tool.
# The job is named "llama-batch1", and the working directory is `distributed_env_deployment/`.
# This will execute the `3_ray_batch_job.py` script on the cluster.
# ray job submit --submission-id llamma-batch1 --working-dir distributed_env_deployment/ -- python 3_ray_batch_job.py

# Performance metric:
# Throughput: 111 inputs per second


"""
Brief Explanation of Ray:
Ray is a distributed computing framework that allows developers to scale their 
Python applications across clusters of machines or GPUs. It provides a simple interface 
for parallelizing tasks and managing resources efficiently. 
Ray makes it easy to distribute both machine learning tasks (such as model training and inference) 
and general workloads. In this code:

Parallelism: Ray allows the function predict_batch to run on multiple GPUs and CPUs, 
which accelerates the processing of large batches of prompts.
Remote Execution: By using @ray.remote, the function can be executed in a distributed environment, 
enabling parallel processing and improving throughput.
"""
