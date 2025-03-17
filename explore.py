import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from common import VAE, get_dataloaders
from sklearn.decomposition import PCA
import ipywidgets as widgets
from IPython.display import display

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "vae_checkpoint.pth"
BATCH_SIZE = 128

# Load model checkpoint.
def load_model() -> VAE:
    model = VAE().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

# Load MNIST test data.
def load_test_data() -> tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    imgs, lbls = [], []
    with torch.no_grad():
        for x, y in loader:
            imgs.append(x)
            lbls.append(y)
    return torch.cat(imgs, dim=0), torch.cat(lbls, dim=0)

model = load_model()
images, labels = load_test_data()

# Encode the entire test set.
def encode_all(x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        mu, _ = model.encode(x.to(DEVICE))
    return mu.detach().cpu().numpy()

latent_codes = encode_all(images)

# Reduce latent codes to 2D via PCA for visualization.
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_codes)

def plot_latent_space() -> None:
    plt.figure(figsize=(6,6))
    sc = plt.scatter(latent_2d[:,0], latent_2d[:,1], c=labels.numpy(), cmap='tab10', alpha=0.7)
    plt.colorbar(sc, ticks=range(10))
    plt.title("Latent Space (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

plot_latent_space()

# Interactive interpolation between two test images.
def interpolate(idx1: int, idx2: int, alpha: float) -> None:
    x1 = images[idx1].unsqueeze(0).to(DEVICE)
    x2 = images[idx2].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)
        z = (1 - alpha) * mu1 + alpha * mu2
        recon = model.decode(z)
    img = recon.detach().cpu().view(28, 28).numpy()
    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap='gray')
    plt.title(f"Interpolation\nalpha = {alpha:.2f}")
    plt.axis('off')
    plt.show()

interp_ui = widgets.interactive(interpolate,
    idx1=widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0),
    idx2=widgets.IntSlider(min=0, max=len(images)-1, step=1, value=1),
    alpha=widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=0.5)
)
display(interp_ui)

# Interactive sampling around a latent code.
def sample_around(idx: int, std: float, num_samples: int) -> None:
    x = images[idx].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu, _ = model.encode(x)
    samples = []
    for _ in range(num_samples):
        noise = torch.randn_like(mu) * std
        z = mu + noise
        recon = model.decode(z)
        samples.append(recon.detach().cpu().view(28, 28).numpy())
    # Plot samples in a square grid.
    n = int(np.ceil(np.sqrt(num_samples)))
    fig, axs = plt.subplots(n, n, figsize=(n*2, n*2))
    for i in range(n*n):
        ax = axs[i//n, i%n]
        if i < num_samples:
            ax.imshow(samples[i], cmap='gray')
        ax.axis('off')
    plt.suptitle(f"Sampling around image {idx} | std = {std}")
    plt.show()

sample_ui = widgets.interactive(sample_around,
    idx=widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0),
    std=widgets.FloatSlider(min=0.1, max=2.0, step=0.1, value=0.5),
    num_samples=widgets.IntSlider(min=1, max=16, step=1, value=9)
)
display(sample_ui)
