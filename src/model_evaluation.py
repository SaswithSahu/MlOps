import pandas as pd
import joblib
import mlflow
from sklearn.metrics import accuracy_score, classification_report

class ModelEvaluation:
    def __init__(self, test_path, model_path):
        self.test_path = test_path
        self.model_path = model_path

    def evaluate(self):
        df = pd.read_csv(self.test_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        model = joblib.load(self.model_path)
        preds = model.predict(X)

        acc = accuracy_score(y, preds)
        report = classification_report(y, preds, output_dict=True)

        with mlflow.start_run():
            mlflow.log_metric("accuracy", acc)

        print(f"Accuracy: {acc}")
        return acc, report

if __name__ == "__main__":
    test_path = "data/processed/test.csv"
    model_path = "models/model.pkl"

    evaluator = ModelEvaluation(test_path, model_path)
    evaluator.evaluate()
