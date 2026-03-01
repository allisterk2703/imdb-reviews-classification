import os

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from model import build_model, train_model


OUTPUT_DIR = "output"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(f"{OUTPUT_DIR}/embeddings.csv")
    X = df.drop(columns=["sentiment"]).values
    y = df["sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = build_model()
    history = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=10,
        batch_size=64,
    )

    with open(f"{OUTPUT_DIR}/training_metrics.txt", "w") as f:
        for m in history:
            f.write(
                f"Epoch {m['epoch']} — loss: {m['loss']:.4f} — "
                f"val_loss: {m['val_loss']:.4f} — val_accuracy: {m['val_accuracy']:.4f}\n"
            )

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze()
        accuracy = ((preds > 0.5) == torch.tensor(y_test).bool()).float().mean().item()

    with open(f"{OUTPUT_DIR}/results.txt", "w") as f:
        f.write(f"Test accuracy: {accuracy:.4f}\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Positive predicted: {(preds > 0.5).sum().item()}\n")
        f.write(f"Negative predicted: {(preds <= 0.5).sum().item()}\n")

    y_pred = (preds > 0.5).int().numpy()
    cm = confusion_matrix(y_test, y_pred)
    labels = ["negative", "positive"]
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    cm_df.to_csv(f"{OUTPUT_DIR}/confusion_matrix.csv")
    with open(f"{OUTPUT_DIR}/confusion_matrix.txt", "w") as f:
        f.write("Confusion matrix (rows = true, columns = predicted)\n\n")
        f.write(cm_df.to_string())
        f.write("\n\n")
        tn, fp, fn, tp = cm.ravel()
        f.write(f"TN (true negative):   {tn}\n")
        f.write(f"FP (false positive):  {fp}\n")
        f.write(f"FN (false negative):  {fn}\n")
        f.write(f"TP (true positive):   {tp}\n")

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Results saved to {OUTPUT_DIR}/results.txt")
    print(f"Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.csv and {OUTPUT_DIR}/confusion_matrix.txt")


if __name__ == "__main__":
    main()
