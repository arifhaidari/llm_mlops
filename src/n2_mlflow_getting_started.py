import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import os

# Set tracking URI
mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Set experiment by name, creating it if it doesn't exist
mlflow.set_experiment('My Experiment')

# Start the run
with mlflow.start_run():
    pass

# Create run with a custom name
run = mlflow.start_run(run_name="custom run")

# Logging parameters
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("batch_size", 32)

num_epochs = 10
mlflow.log_param('num_epochs', num_epochs)

# Logging metrics for each epoch
for epoch in range(num_epochs):
    mlflow.log_metric("accuracy", np.random.random(), step=epoch)
    mlflow.log_metric("loss", np.random.random(), step=epoch)

# Logging a time-series metric
for t in range(100):
    metric_value = np.sin(t * np.pi / 50)
    mlflow.log_metric("time_series_metric", metric_value, step=t)

# =================
# Logging datasets
with open("data/dataset.csv", "w") as f:
     f.write("x,y\n")
     for x in range(100):
          f.write(f"{x},{x * 2}\n")

mlflow.log_artifact("data/dataset.csv", "data")

# =================

# saving different types of artifacts

# Generate a confusion matrix
confusion_matrix = np.random.randint(0, 100, size=(5, 5))  # 5x5 matrix

labels = ["Class A", "Class B", "Class C", "Class D", "Class E"]
df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

# Plot confusion matrix using Plotly Express
fig = px.imshow(df_cm, text_auto=True, labels=dict(x="Predicted Label", y="True Label"), x=labels, y=labels, title="Confusion Matrix")

# My artifacts folder
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Define the file path for saving the HTML file
html_file = os.path.join(output_folder, "confusion_matrix.html")

# Save the figure as an HTML file
fig.write_html(html_file)

# Log the HTML file with MLflow
mlflow.log_artifact(html_file)

# ===================
# Logging models 
from transformers import AutoModelForSeq2SeqLM

# Initialize a model from Hugging Face Transformers
model = AutoModelForSeq2SeqLM.from_pretrained("TheFuzzyScientist/T5-base_Amazon-product-reviews")

# Log the model in MLflow
mlflow.pytorch.log_model(model, "transformer_model")

mlflow.end_run()

# code ends here
print('end of the code')
