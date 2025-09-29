import pandas as pd
import tempfile
import os
import joblib
from src.3_model_training import ModelTrainer

def test_model_training_creates_model(tmp_path):
    # Create a small sample training CSV
    train_csv = tmp_path / "train.csv"
    df = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 6.2, 5.9],
        "sepal_width": [3.5, 3.0, 3.4, 3.0],
        "petal_length": [1.4, 1.4, 5.4, 5.1],
        "petal_width": [0.2, 0.2, 2.3, 1.8],
        "target": [0, 0, 2, 2]
    })
    df.to_csv(train_csv, index=False)

    # Set parameters
    params = {
        "model": "random_forest",
        "n_estimators": 10,
        "max_depth": 2
    }

    # Path to save model
    model_path = tmp_path / "model.pkl"

    # Train model
    trainer = ModelTrainer(str(train_csv), str(model_path), params)
    trainer.train()

    # Check if model file is created
    assert os.path.exists(model_path)

    # Load model and make prediction
    model = joblib.load(model_path)
    sample_input = df.drop("target", axis=1).iloc[:1]
    pred = model.predict(sample_input)
    assert pred.shape[0] == 1
