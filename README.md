# LLM with MLOPs

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
