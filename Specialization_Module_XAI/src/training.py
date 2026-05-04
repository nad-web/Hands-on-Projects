import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader,
                epochs=50, lr=1e-3, patience=5,
                weight_decay=1e-4, device='cpu'):
    """Train CNN with early stopping and cosine annealing."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss, patience_counter, best_state = float('inf'), 0, None

    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += criterion(model(x.to(device)), y.to(device)).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[Early Stop] Epoch {epoch+1} | Val Loss: {val_loss:.4f}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return model
