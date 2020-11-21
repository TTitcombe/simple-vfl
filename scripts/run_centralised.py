"""
Train a LogisticRegression model
on the Titanic dataset
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.feature_engineering import get_datasets, scale


x_train, y_train, x_test = get_datasets(
    pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
)
x_train, x_test = scale(x_train, x_test)

lr = LogisticRegression()
lr.fit(x_train, y_train)

pred_train = lr.predict(x_train)
pred_test = lr.predict(x_test)

train_acc = accuracy_score(pred_train, y_train)
print(f"Train accuracy: {100*train_acc:.3f}%")
