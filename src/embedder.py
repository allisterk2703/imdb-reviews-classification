import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_NAME = "distilbert-base-uncased"


def load_encoder(model_name: str = DEFAULT_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def get_embeddings(
    texts,
    tokenizer,
    model,
    batch_size: int = 32,
    max_length: int = 128,
):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


def generate_and_save_embeddings(
    input_path: str = "input/imdb_reviews.csv",
    output_path: str = "output/embeddings.csv",
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> np.ndarray:
    df = pd.read_csv(input_path)
    texts = df["review"].tolist()
    y = (df["sentiment"] == "positive").astype(int).values

    tokenizer, model = load_encoder(model_name)
    X = get_embeddings(texts, tokenizer, model, batch_size=batch_size)

    out_df = pd.DataFrame(X)
    out_df["sentiment"] = y
    out_df.to_csv(output_path, index=False)

    return X, y


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    generate_and_save_embeddings()
    print("Embeddings saved to output/embeddings.csv")
