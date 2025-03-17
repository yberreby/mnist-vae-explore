import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from common import VAE, loss_fn, get_dataloaders, IMG_SIZE, HIDDEN_DIM, LATENT_DIM

# Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 15
BASE_LR = 1e-3  # baseline LR for batch size 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "vae_checkpoint.pth"

# Display hyperparameters clearly
print("Training Hyperparameters:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Base LR: {BASE_LR}")
print(f"  Hidden dim: {HIDDEN_DIM}")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Device: {DEVICE}")

# Linear warmup scheduler: ramp LR from 0 to 1 over warmup_steps
def lr_lambda(step: int) -> float:
    warmup_steps = 500
    return float(step) / float(max(1, warmup_steps)) if step < warmup_steps else 1.0

def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            recon, mu, logvar = model(x)
            total_loss += loss_fn(recon, x, mu, logvar).item() * x.size(0)
    return total_loss / len(loader.dataset)

def train() -> None:
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)
    model = VAE().to(DEVICE)

    # Show model summary using torchinfo
    summary(model, input_size=(BATCH_SIZE, IMG_SIZE))

    # Linear scaling of LR with batch size.
    optimizer = AdamW(model.parameters(), lr=BASE_LR * (BATCH_SIZE / 128))
    scheduler = LambdaLR(optimizer, lr_lambda)

    writer = SummaryWriter("runs/vae_experiment")
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = loss_fn(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * x.size(0)
            writer.add_scalar("Loss/Train_Step", loss.item(), global_step)
            global_step += 1

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = evaluate(model, val_loader)
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save the checkpoint and hyperparameters
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "base_lr": BASE_LR,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM
        }
    }, CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")
    writer.close()

if __name__ == '__main__':
    train()
