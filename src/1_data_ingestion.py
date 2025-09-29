import pandas as pd
from sklearn.datasets import load_iris
import os

class DataIngestion:
    def __init__(self, output_path):
        self.output_path = output_path

    def load_data(self):
        iris = load_iris(as_frame=True)
        df = iris.frame
        df["target"] = iris.target

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Iris dataset saved to {self.output_path}")
        return df

if __name__ == "__main__":
    out_path = "data/raw/iris.csv"
    ingestion = DataIngestion(out_path)
    ingestion.load_data()
