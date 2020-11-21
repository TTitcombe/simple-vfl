"""
Functions to create a useable dataset from the raw titanic data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _bin_age(value):
    if np.isnan(value):
        return "Unknown"
    elif value <= 10:
        return "Child"
    elif value <= 40:
        return "Adult"
    else:
        return "Elderly"


def create_features(df):
    df["Age"] = df["Age"].apply(_bin_age)

    df["Cabin"] = df["Cabin"].apply(lambda x: x[0])

    df["Title"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )

    df["Title"] = df["Title"].replace("Mlle", "Miss")
    df["Title"] = df["Title"].replace("Ms", "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")

    df = pd.get_dummies(
        df, columns=["Sex", "Pclass", "Embarked", "Title", "Cabin", "Age"]
    )
    df = df.drop(columns=["PassengerId", "Name", "Ticket",], axis=1,)

    return df


def clean(df):
    df = df.loc[df.Embarked.notnull()]  # Only a few Embarked NaN, so remove
    df = df.loc[df.Fare.notnull()]  # Only a few Fare NaN, so remove
    df.loc[df.Cabin.isna(), "Cabin"] = "Unknown"

    return df


def get_datasets(df, df_test):
    # Combine to get correct number of dummy features
    df_combined = df.append(df_test)
    df_combined = clean(df_combined)
    df_combined = create_features(df_combined)

    df = df_combined.loc[df_combined.Survived.notnull()]
    df_test = df_combined.loc[df_combined.Survived.isna()]

    train_y = df["Survived"].values
    train_x = df.drop("Survived", axis=1)

    test_x = df_test.drop("Survived", axis=1)

    return train_x, train_y, test_x


def scale(train, test):
    s = StandardScaler()
    train = s.fit_transform(train)
    test = s.transform(test)

    return train, test


def partition_data(df):
    partition_1_keywords = ("Parch", "Cabin", "Pclass")
    partition_1_columns = []

    for kw in partition_1_keywords:
        partition_1_columns.extend([c for c in df.columns if kw in c])

    partition_2_keywords = ("Sex", "Title")
    partition_2_columns = []

    for kw in partition_2_keywords:
        partition_2_columns.extend([c for c in df.columns if kw in c])

    partition_3_columns = list(
        set(df.columns) - set(partition_1_columns) - set(partition_2_columns)
    )

    return df[partition_1_columns], df[partition_2_columns], df[partition_3_columns]
