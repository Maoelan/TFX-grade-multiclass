docker build -t grade-prediction-model-tf-serving .

docker run -p 8080:8501 grade-prediction-model-tf-serving