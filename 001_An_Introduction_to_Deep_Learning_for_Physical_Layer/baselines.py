import itertools
from typing import Iterable, List

import numpy as np
import scipy.special
import torch


def uncoded_bpsk_bler(k: int, snr_db_range: Iterable[float]) -> List[float]:
    out: List[float] = []
    for snr_db in snr_db_range:
        snr_lin = 10.0 ** (snr_db / 10.0)
        ber = 0.5 * scipy.special.erfc(np.sqrt(snr_lin))
        out.append(1.0 - (1.0 - ber) ** k)
    return out


def hamming_74_hard_bler(snr_db_range: Iterable[float]) -> List[float]:
    n, k = 7, 4
    r = k / n
    out: List[float] = []
    for snr_db in snr_db_range:
        snr_lin = 10.0 ** (snr_db / 10.0)
        p = 0.5 * scipy.special.erfc(np.sqrt(r * snr_lin))
        prob_success = (1.0 - p) ** 7 + 7 * p * (1.0 - p) ** 6
        out.append(1.0 - prob_success)
    return out


def hamming_74_mld_bler(
    snr_db_range: Iterable[float],
    device: torch.device,
    num_samples: int,
    batch_size: int,
) -> List[float]:
    n, k = 7, 4
    r = k / n
    g = np.array(
        [
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ]
    )
    msgs = np.array(list(itertools.product([0, 1], repeat=4)))
    codewords = np.dot(msgs, g) % 2
    tx_symbols = 1.0 - 2.0 * codewords
    tx_tensor = torch.tensor(tx_symbols, dtype=torch.float32, device=device)

    out: List[float] = []
    for snr_db in snr_db_range:
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_std = np.sqrt(1.0 / (2.0 * r * snr_lin))
        errors = 0
        for _ in range(num_samples // batch_size):
            labels = torch.randint(0, 16, (batch_size,), device=device)
            tx_batch = tx_tensor[labels]
            rx_batch = tx_batch + torch.randn_like(tx_batch) * noise_std
            dists = torch.sum((rx_batch.unsqueeze(1) - tx_tensor.unsqueeze(0)) ** 2, dim=2)
            preds = torch.argmin(dists, dim=1)
            errors += (preds != labels).sum().item()
        out.append(errors / num_samples)
    return out
