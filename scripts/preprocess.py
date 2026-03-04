import pandas as pd
from src.preprocess import extract_embeddings, save_embeddings

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

X_train = extract_embeddings(train["text"])
X_test = extract_embeddings(test["text"])

save_embeddings("data/processed/train_emb.npy", X_train) 
save_embeddings("data/processed/test_emb.npy", X_test)
