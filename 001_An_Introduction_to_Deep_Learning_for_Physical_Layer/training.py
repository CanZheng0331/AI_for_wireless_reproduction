import copy
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Autoencoder


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 512
    iterations_per_epoch: int = 200
    eval_iterations: int = 50
    train_ebno_db: float = 7.0
    lr: float = 0.001
    test_num_samples: int = 1_000_000
    test_batch_size: int = 50_000


def train_autoencoder(
    n: int,
    k: int,
    device: torch.device,
    cfg: TrainConfig,
    norm_type: str = "fixed",
) -> Autoencoder:
    m = 2**k
    r = k / n
    model = Autoencoder(m, n, norm_type=norm_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    train_ebno_linear = 10.0 ** (cfg.train_ebno_db / 10.0)
    noise_std = np.sqrt(1.0 / (2.0 * r * train_ebno_linear))

    best_state_dict = copy.deepcopy(model.state_dict())
    best_test_loss = float("inf")
    best_epoch = 0

    print(f"Training Autoencoder ({n},{k}) (M={m}, norm={norm_type})...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for _ in range(cfg.iterations_per_epoch):
            labels = torch.randint(0, m, (cfg.batch_size,), device=device)
            x = F.one_hot(labels, num_classes=m).float()
            optimizer.zero_grad()
            logits = model(x, noise_std)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / cfg.iterations_per_epoch
        if epoch % 10 == 0:
            model.eval()
            test_loss_sum = 0.0
            with torch.no_grad():
                for _ in range(cfg.eval_iterations):
                    labels = torch.randint(0, m, (cfg.batch_size,), device=device)
                    x = F.one_hot(labels, num_classes=m).float()
                    tx = model.encode(x)
                    rx = tx + torch.randn_like(tx) * noise_std
                    logits = model.decode(rx)
                    test_loss_sum += criterion(logits, labels).item()

            avg_test_loss = test_loss_sum / cfg.eval_iterations
            print(
                f"Epoch [{epoch}/{cfg.epochs}] "
                f"Training Loss: {avg_train_loss:.6f} | Testing Loss: {avg_test_loss:.6f}"
            )
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

    if best_epoch == 0:
        best_state_dict = copy.deepcopy(model.state_dict())
        best_epoch = cfg.epochs

    model.load_state_dict(best_state_dict)
    print(f"Best model: epoch={best_epoch}, testing_loss={best_test_loss:.6f}")
    return model


def test_autoencoder(
    model: Autoencoder,
    n: int,
    k: int,
    device: torch.device,
    snr_db_range: Iterable[float],
    num_samples: int,
    batch_size: int,
) -> List[float]:
    m = 2**k
    r = k / n
    model.eval()
    bler_list: List[float] = []
    with torch.no_grad():
        for snr_db in snr_db_range:
            snr_lin = 10.0 ** (snr_db / 10.0)
            noise_std = np.sqrt(1.0 / (2.0 * r * snr_lin))
            errors = 0
            for _ in range(num_samples // batch_size):
                labels = torch.randint(0, m, (batch_size,), device=device)
                x = F.one_hot(labels, num_classes=m).float()
                tx = model.encode(x)
                rx = tx + torch.randn_like(tx) * noise_std
                logits = model.decode(rx)
                preds = torch.argmax(logits, dim=1)
                errors += (preds != labels).sum().item()
            bler_list.append(errors / num_samples)
    return bler_list


def get_constellation(model: Autoencoder, k: int, device: torch.device) -> np.ndarray:
    m = 2**k
    model.eval()
    with torch.no_grad():
        labels = torch.arange(m, device=device)
        x = F.one_hot(labels, num_classes=m).float()
        constellation = model.encode(x).cpu().numpy()
    return constellation


def sample_noisy_rx_for_tsne(
    model: Autoencoder,
    n: int,
    k: int,
    device: torch.device,
    snr_db: float = 7.0,
    num_samples_per_msg: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    m = 2**k
    r = k / n
    model.eval()
    with torch.no_grad():
        labels = torch.arange(m, device=device).repeat_interleave(num_samples_per_msg)
        x = F.one_hot(labels, num_classes=m).float()
        tx = model.encode(x)
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_std = np.sqrt(1.0 / (2.0 * r * snr_lin))
        rx = tx + torch.randn_like(tx) * noise_std
    return rx.cpu().numpy(), labels.cpu().numpy()
