from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

from baselines import hamming_74_hard_bler, hamming_74_mld_bler, uncoded_bpsk_bler
from training import (
    TrainConfig,
    get_constellation,
    sample_noisy_rx_for_tsne,
    test_autoencoder,
    train_autoencoder,
)
from utils import get_device, set_seed


def run_figure_3a(
    device: torch.device,
    train_cfg: TrainConfig,
    snr_start: int = -4,
    snr_end: int = 9,
    snr_step: int = 1,
    mld_num_samples: int = 1_000_000,
    mld_batch_size: int = 50_000,
) -> Dict[str, object]:
    snr_range_a = np.arange(snr_start, snr_end, snr_step)
    bler_uncoded_44 = uncoded_bpsk_bler(4, snr_range_a)
    bler_ham_hard = hamming_74_hard_bler(snr_range_a)
    bler_ham_mld = hamming_74_mld_bler(
        snr_range_a,
        device=device,
        num_samples=mld_num_samples,
        batch_size=mld_batch_size,
    )

    best_model_74 = train_autoencoder(n=7, k=4, device=device, cfg=train_cfg, norm_type="fixed")
    bler_ae_74 = test_autoencoder(
        best_model_74,
        n=7,
        k=4,
        device=device,
        snr_db_range=snr_range_a,
        num_samples=train_cfg.test_num_samples,
        batch_size=train_cfg.test_batch_size,
    )
    return {
        "snr_range_a": snr_range_a,
        "bler_uncoded_44": bler_uncoded_44,
        "bler_ham_hard": bler_ham_hard,
        "bler_ham_mld": bler_ham_mld,
        "bler_ae_74": bler_ae_74,
        "best_model_74": best_model_74,
    }


def run_figure_3b(
    device: torch.device,
    train_cfg_default: TrainConfig,
    train_cfg_88: TrainConfig,
    snr_start: int = -2,
    snr_end: int = 11,
    snr_step: int = 1,
) -> Dict[str, object]:
    snr_range_b = np.arange(snr_start, snr_end, snr_step)
    bler_uncoded_88 = uncoded_bpsk_bler(8, snr_range_b)
    bler_uncoded_22 = uncoded_bpsk_bler(2, snr_range_b)

    best_model_88 = train_autoencoder(n=8, k=8, device=device, cfg=train_cfg_88, norm_type="fixed")
    bler_ae_88 = test_autoencoder(
        best_model_88,
        n=8,
        k=8,
        device=device,
        snr_db_range=snr_range_b,
        num_samples=train_cfg_88.test_num_samples,
        batch_size=train_cfg_88.test_batch_size,
    )

    best_model_22 = train_autoencoder(n=2, k=2, device=device, cfg=train_cfg_default, norm_type="fixed")
    bler_ae_22 = test_autoencoder(
        best_model_22,
        n=2,
        k=2,
        device=device,
        snr_db_range=snr_range_b,
        num_samples=train_cfg_default.test_num_samples,
        batch_size=train_cfg_default.test_batch_size,
    )
    return {
        "snr_range_b": snr_range_b,
        "bler_uncoded_88": bler_uncoded_88,
        "bler_uncoded_22": bler_uncoded_22,
        "bler_ae_88": bler_ae_88,
        "bler_ae_22": bler_ae_22,
        "best_model_88": best_model_88,
        "best_model_22": best_model_22,
    }


def plot_figure_3(a_data: Dict[str, object], b_data: Dict[str, object]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.semilogy(a_data["snr_range_a"], a_data["bler_uncoded_44"], "k-", label="Uncoded BPSK (4,4)")
    ax1.semilogy(
        a_data["snr_range_a"], a_data["bler_ham_hard"], "b-.", label="Hamming (7,4) Hard Decision"
    )
    ax1.semilogy(a_data["snr_range_a"], a_data["bler_ae_74"], "r.", markersize=10, label="Autoencoder (7,4)")
    ax1.semilogy(a_data["snr_range_a"], a_data["bler_ham_mld"], "b--", label="Hamming (7,4) MLD")
    ax1.set_xlim([-4, 8])
    ax1.set_ylim([1e-5, 1])
    ax1.set_xlabel("$E_b/N_0$ [dB]")
    ax1.set_ylabel("Block error rate")
    ax1.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray", alpha=0.5)
    ax1.legend(loc="lower left")
    ax1.set_title("(a)")

    ax2.semilogy(b_data["snr_range_b"], b_data["bler_uncoded_88"], "k-", label="Uncoded BPSK (8,8)")
    ax2.semilogy(
        b_data["snr_range_b"], b_data["bler_ae_88"], "r-o", markerfacecolor="none", label="Autoencoder (8,8)"
    )
    ax2.semilogy(b_data["snr_range_b"], b_data["bler_uncoded_22"], "k--", label="Uncoded BPSK (2,2)")
    ax2.semilogy(
        b_data["snr_range_b"], b_data["bler_ae_22"], "r--s", markerfacecolor="none", label="Autoencoder (2,2)"
    )
    ax2.set_xlim([-2, 10])
    ax2.set_ylim([1e-5, 1])
    ax2.set_xlabel("$E_b/N_0$ [dB]")
    ax2.set_ylabel("Block error rate")
    ax2.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray", alpha=0.5)
    ax2.legend(loc="lower left")
    ax2.set_title("(b)")

    plt.tight_layout()
    plt.show()


def run_figure_4(
    device: torch.device,
    train_cfg_default: TrainConfig,
    train_cfg_fig4: TrainConfig | None = None,
    tsne_perplexity: float = 30.0,
    tsne_random_state: int = 42,
    tsne_samples_per_msg: int = 60,
    tsne_snr_db: float = 7.0,
) -> Dict[str, object]:
    cfg_fig4 = train_cfg_fig4 or TrainConfig(
        epochs=150,
        batch_size=train_cfg_default.batch_size,
        iterations_per_epoch=100,
        eval_iterations=train_cfg_default.eval_iterations,
        train_ebno_db=train_cfg_default.train_ebno_db,
        lr=0.005,
        test_num_samples=train_cfg_default.test_num_samples,
        test_batch_size=train_cfg_default.test_batch_size,
    )

    model_a = train_autoencoder(n=2, k=2, device=device, cfg=cfg_fig4, norm_type="fixed")
    const_a = get_constellation(model_a, k=2, device=device)

    model_b = train_autoencoder(n=2, k=4, device=device, cfg=cfg_fig4, norm_type="fixed")
    const_b = get_constellation(model_b, k=4, device=device)

    model_c = train_autoencoder(n=2, k=4, device=device, cfg=cfg_fig4, norm_type="average")
    const_c = get_constellation(model_c, k=4, device=device)

    model_d = train_autoencoder(n=7, k=4, device=device, cfg=cfg_fig4, norm_type="fixed")
    rx_d_np, labels_d_np = sample_noisy_rx_for_tsne(
        model=model_d,
        n=7,
        k=4,
        device=device,
        snr_db=tsne_snr_db,
        num_samples_per_msg=tsne_samples_per_msg,
    )

    print("Running t-SNE for Figure 4(d)...")
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=tsne_random_state)
    rx_tsne = tsne.fit_transform(rx_d_np)

    return {
        "const_a": const_a,
        "const_b": const_b,
        "const_c": const_c,
        "rx_tsne": rx_tsne,
        "labels_d": labels_d_np,
        "model_a": model_a,
        "model_b": model_b,
        "model_c": model_c,
        "model_d": model_d,
    }


def plot_figure_4(fig4_data: Dict[str, object]) -> None:
    const_a = fig4_data["const_a"]
    const_b = fig4_data["const_b"]
    const_c = fig4_data["const_c"]
    rx_tsne = fig4_data["rx_tsne"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    def format_plot(ax, title: str) -> None:
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.grid(True, linestyle="-", linewidth=0.5, color="lightgray")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(title, fontsize=12)

    axs[0, 0].plot(const_a[:, 0], const_a[:, 1], "ko", markersize=8)
    format_plot(axs[0, 0], "(a)")

    axs[0, 1].plot(const_b[:, 0], const_b[:, 1], "ko", markersize=8)
    format_plot(axs[0, 1], "(b)")

    axs[1, 0].plot(const_c[:, 0], const_c[:, 1], "ko", markersize=8)
    format_plot(axs[1, 0], "(c)")

    axs[1, 1].plot(rx_tsne[:, 0], rx_tsne[:, 1], "k.", markersize=4)
    tsne_max = np.max(np.abs(rx_tsne))
    axs[1, 1].set_xlim([-tsne_max * 1.1, tsne_max * 1.1])
    axs[1, 1].set_ylim([-tsne_max * 1.1, tsne_max * 1.1])
    axs[1, 1].grid(True, linestyle="-", linewidth=0.5, color="lightgray")
    axs[1, 1].set_aspect("equal", adjustable="box")
    axs[1, 1].set_xlabel("(d)", fontsize=12)
    axs[1, 1].set_xticklabels([])
    axs[1, 1].set_yticklabels([])

    plt.show()


__all__ = [
    "TrainConfig",
    "set_seed",
    "get_device",
    "run_figure_3a",
    "run_figure_3b",
    "plot_figure_3",
    "run_figure_4",
    "plot_figure_4",
]
