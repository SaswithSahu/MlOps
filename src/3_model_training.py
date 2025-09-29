import pandas as pd
import sys
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml
import os
import subprocess  # <-- added for git commit

class ModelTrainer:
    def __init__(self, train_path, model_path, params):
        self.train_path = train_path
        self.model_path = model_path
        self.params = params

    def train(self):
        df = pd.read_csv(self.train_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        if self.params["model"] == "logistic":
            model = LogisticRegression(max_iter=200)
        else:
            model = RandomForestClassifier(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                random_state=42
            )

        # -------------------------------
        # Get current git commit hash dynamically
        # -------------------------------
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()

        mlflow.set_experiment("Iris Classification")
        with mlflow.start_run():
            model.fit(X, y)

            # Log parameters
            mlflow.log_param("model", self.params["model"])
            if self.params["model"] == "random_forest":
                mlflow.log_param("n_estimators", self.params["n_estimators"])
                mlflow.log_param("max_depth", self.params["max_depth"])

            # Log git/DVC commit dynamically
            mlflow.log_param("dvc_commit", git_commit)

            # Log model artifact
            mlflow.sklearn.log_model(model, "model")

        # Save local model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        print(f"Model saved at {self.model_path}")

if __name__ == "__main__":
    train_path = "data/processed/train.csv"
    model_path = "models/model.pkl"
    params_file = "params.yaml"

    params = yaml.safe_load(open(params_file))["train"]

    trainer = ModelTrainer(train_path, model_path, params)
    trainer.train()
