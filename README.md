# LLM and Transformers with MLOPs (Under Development)

in this project I will build the model and then use all the MLOPs methods and tools to deploy and scale the model.

table of Content:

- Pre-deployment consideration
- Model Management
- MLOPs
- Data Management
- Model Deployment
- Model Monitoring
- MLFLOW

---

the content goes here
Mlflow is library-agnostic

integrate mysql or sqlite for backend and mlflow to store metrics

for storing the artifacts I will use the S3 in production and local storage in the start

to run mlflow for the first time:
mlflow server --backend-store-uri sqlite:///metrics_store/mlflow.db --default-artifact-root /Users/arifmoazy/AI/Gen_AI/llm_mlops/artifact_store --host 0.0.0.0 --port 5000

I did install the transformers and pytorch
but for doing:
Log the model in MLflow
mlflow.pytorch.log_model(model, "transformer_model")
I needed to install:
pip install torch torchvision torchaudio
since I am using macbook

Training models with MLFLOW:
train a News classification and categorisation with transformers

Creating batches dynamically to tokenize (transformers)

batch with sorting

quantization for model efficiency:
used CTranslate2, it does a lot of quantization technique (add more details in this part)
using this we can structure the weight of the model to be effiently used by different logical flow during inference.

LORA:
it is very good to customize the pre-trained models and preserve the knowledge and train with the new knowledge.

transformers:
add more information and explain for transformers

transformers with pipelines, without pipelines and with ctranslate's generator

transformers with optimisation and VLLM

deploying on distributed system:
using the RabbitMQ message broker and install it on different system.
the interaction with rabbitmq server would be done with pika

scaling up the deployment using Ray:
scaling from a single machine to a large cluster effectively.

Disclaimer:
part of this code is inspired by different sources such as tutorials, stackoverflow, udemy, Medium and github. the credit of it goes to respective source. the purpose of these gathering was learning, reference and practice purposes.
