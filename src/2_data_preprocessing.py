import pandas as pd
from sklearn.model_selection import train_test_split
import os

class DataPreprocessing:
    def __init__(self, input_path, train_path, test_path, test_size=0.2):
        self.input_path = input_path
        self.train_path = train_path
        self.test_path = test_path
        self.test_size = test_size

    def preprocess(self):
        df = pd.read_csv(self.input_path)

        # No missing values in Iris, but let's ensure
        df = df.dropna()

        train, test = train_test_split(df, test_size=self.test_size, random_state=42, stratify=df["target"])

        os.makedirs(os.path.dirname(self.train_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.test_path), exist_ok=True)

        train.to_csv(self.train_path, index=False)
        test.to_csv(self.test_path, index=False)
        print(f"Train/Test saved: {self.train_path}, {self.test_path}")

if __name__ == "__main__":
    input_path = "data/raw/iris.csv"
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    preprocessing = DataPreprocessing(input_path, train_path, test_path)
    preprocessing.preprocess()
