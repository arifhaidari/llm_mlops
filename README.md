# LLM and Transformers with MLOPs Boilerplate

This project explores deploying and scaling transformer-based models using modern MLOps techniques. The goal is to create a comprehensive learning environment and boilerplate for future projects involving large language models (LLMs), transformers, and MLOps best practices.

## Table of Contents

- [Pre-deployment Considerations](#pre-deployment-considerations)
- [Model Management](#model-management)
- [MLOps](#mlops)
- [Data Management](#data-management)
- [Model Deployment](#model-deployment)
- [Model Monitoring](#model-monitoring)
- [MLflow](#mlflow)
- [Optimizations](#optimizations)
- [Distributed Systems and Scaling](#distributed-systems-and-scaling)
- [Disclaimer](#disclaimer)

---

## Pre-deployment Considerations

Before deploying any machine learning model, several considerations must be addressed:

1. **Hardware & Resource Management**: Efficient use of CPU/GPU resources using frameworks like `ray` for distributed computing.
2. **Tokenization**: Utilizing libraries like HuggingFace's `transformers` to tokenize text data effectively, ensuring prompt processing.
3. **Quantization**: Techniques like model quantization with `CTranslate2` to reduce model size and inference latency.

## Model Management

The project leverages the `transformers` library for model management:

- HuggingFace’s **AutoTokenizer** and **AutoModelForCausalLM** for transformer-based models, including pre-trained models like `TinyLlama-1.1B-Chat-v1.0`.
- Dynamic batching for efficient token generation during inference.
- Custom prompts created using the tokenizer’s chat templates.

## MLOps

**MLOps** tools streamline the deployment, monitoring, and maintenance of machine learning models. Key components include:

- **MLflow** for model tracking, experiment logging, and artifact management.
- Model artifacts are stored locally during development and in **AWS S3** for production environments.

### MLflow Configuration

To configure MLflow with a backend store and artifact root:

```bash
mlflow server --backend-store-uri sqlite:///metrics_store/mlflow.db --default-artifact-root /path/to/artifacts --host 0.0.0.0 --port 5000
```

You can also log models in MLflow using `torch`:

```python
mlflow.pytorch.log_model(model, "transformer_model")
```

## Data Management

- **Tokenization & Batching**: Using `transformers`, we efficiently tokenize large batches of text data, ensuring minimal latency. Dynamic batching optimizes the tokenization process.
- **Message Queuing with RabbitMQ**: By utilizing **RabbitMQ** as a message broker, we can distribute tasks across different systems. The `pika` library is used to communicate with RabbitMQ, ensuring smooth interaction between components.

## Model Deployment

- **Transformers Pipelines**: Using HuggingFace’s `pipeline` function allows quick deployment with pre-built models.
- **Without Pipelines**: Custom deployments are implemented using the `AutoModelForCausalLM` and `generate()` methods for flexible inference handling.
- **CTranslate2 Generator**: This framework enables quantization and efficient deployment of models, enhancing speed and reducing resource consumption during inference.

## Model Monitoring

**MLflow** is leveraged to track metrics such as model latency, throughput, and accuracy:

- **Backend Storage**: SQLite for development; MySQL or other relational databases for production.
- **Metrics Logging**: Key performance indicators (KPIs) such as latency, throughput, and model performance are logged for analysis and model improvements.

## Optimizations

- **CTranslate2 Quantization**: By using CTranslate2, models are quantized, reducing model size and improving inference efficiency by leveraging both CPU and GPU resources.
- **LoRA (Low-Rank Adaptation)**: LoRA is used to fine-tune pre-trained models with minimal computation, allowing new knowledge to be integrated without forgetting existing knowledge.

## Distributed Systems and Scaling

For scaling and distributed deployments, we used:

- **RabbitMQ**: As the backbone for message queuing and distributed task management, ensuring smooth communication between multiple systems. The `pika` library facilitates interaction with RabbitMQ.
- **Ray**: For scaling computations from a single machine to a large distributed cluster, `ray` helps in managing resources (CPUs, GPUs) efficiently. By deploying on multiple nodes, the model’s throughput was significantly improved.

Example of deploying a model prediction job using Ray:

```python
ray.init(address="auto")

@ray.remote
def predict_batch():
    # Consume from RabbitMQ, generate predictions, and produce back to RabbitMQ
    ...

ray.get(predict_batch.remote())
```

This process results in a **throughput of 111 inputs/s**, improving upon traditional methods significantly.

## Disclaimer

This project is inspired by various sources including tutorials, StackOverflow discussions, Udemy courses, Medium articles, and GitHub repositories. The purpose is to learn, reference, and practice MLOps and distributed model deployment.

---

This repository serves as a learning platform and boilerplate for deploying and scaling LLMs with MLOps techniques. Each section integrates real-world practices and tools to ensure future projects can build upon this foundation.
