import pandas as pd
import os
from src.data_preprocessing import DataPreprocessing

def test_preprocessing_creates_files(tmp_path):
    # Create a small sample CSV
    sample_csv = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 6.2, 5.9],
        "sepal_width": [3.5, 3.0, 3.4, 3.0],
        "petal_length": [1.4, 1.4, 5.4, 5.1],
        "petal_width": [0.2, 0.2, 2.3, 1.8],
        "target": [0, 0, 2, 2]
    })
    df.to_csv(sample_csv, index=False)

    # Paths for train/test
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    # Run preprocessing
    prep = DataPreprocessing(str(sample_csv), str(train_path), str(test_path), test_size=0.5)
    prep.preprocess()

    # Check if files are created
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # Check if rows split correctly
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert len(train_df) + len(test_df) == 4
    assert "target" in train_df.columns
    assert "target" in test_df.columns
