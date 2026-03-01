import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def build_model(
    input_dim: int = 768,
    hidden_dim: int = 512,
    dropout: float = 0.3,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


def train_model(
    model: nn.Module,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs: int = 10,
    batch_size: int = 64,
):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    history = []
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            preds_test = model(X_test_t).squeeze()
            test_loss = criterion(preds_test, y_test_t).item()
            accuracy = ((preds_test > 0.5) == y_test_t.bool()).float().mean().item()

        scheduler.step(test_loss)

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": total_loss / len(train_loader),
            "val_loss": test_loss,
            "val_accuracy": accuracy,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}/{epochs} — loss: {epoch_metrics['loss']:.4f} — "
            f"val_loss: {test_loss:.4f} — val_accuracy: {accuracy:.4f}"
        )

    model.load_state_dict(best_weights)

    return history
