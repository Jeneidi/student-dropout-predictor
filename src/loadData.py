import pandas as pd

def load_dataset():
    path = "data/student-mat.csv"
    df = pd.read_csv(path, sep=";")
    return df

if __name__ == "__main__":
    df = load_dataset()
    print("Dataset shape:", df.shape)
    print(df.head())

