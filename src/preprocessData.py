import pandas as pd
from loadData import load_dataset

def preprocess_data():
    df = load_dataset()
    df["dropout_risk"] = df["G3"].apply(lambda x: 1 if x < 10 else 0)
    df = df.drop(columns=["G1", "G2", "G3"])
    df = pd.get_dummies(df, drop_first=True)

    return df

if __name__ == "__main__":
    processed_df = preprocess_data()
    print("Processed shape:", processed_df.shape)
    print(processed_df.head())
