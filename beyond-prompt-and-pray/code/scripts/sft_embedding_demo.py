"""Demo: general-purpose low-rank supervised embedding SFT + GMS Mode-B bridge.

Exercises knowlytix.embedding (API 1) on the regulatory {message, regulatory}
paraphrase file, compares rotation vs full and sweeps the drift constraint, then
shows the exported vectors loading into GMS via EmbeddingConfig Mode B (API 2).

NOTE: the val split here is over paraphrases (NOT seed-grouped), so val accuracy
is optimistic — this validates the API, not generalization. The honest
seed-grouped generalization number lives in the GMS cap path
(scripts/kfold_seedgrouped_adapter.py).

    python scripts/sft_embedding_demo.py
"""

from pathlib import Path

import torch

from knowlytix.embedding import EmbeddingSFTConfig, finetune_embedding

AUG = "data/training/regulatory_doe_augmented.jsonl"
D = 256


def run(mode, drift_weight, rank=1, epochs=80):
    cfg = EmbeddingSFTConfig(rank=rank, mode=mode, base_dim=D, drift_weight=drift_weight,
                             epochs=epochs, val_split=0.25, seed=0)
    ft = finetune_embedding(AUG, cfg, text_col="message", label_col="regulatory")
    M = ft.adapter.as_matrix().cpu()
    drift = (M - torch.eye(M.shape[0])).norm().item()
    return ft, drift


def main() -> int:
    print("=== rotation vs full (rank=1, drift_weight=1.0) ===")
    for mode in ("rotation", "full"):
        ft, drift = run(mode, 1.0)
        print(f"  {mode:8s}: val_acc={ft.val_accuracy:.3f}  ||T-I||_F={drift:.3f}  "
              f"classes={ft.label_order}")

    print("\n=== drift_weight sweep (full, rank=1): higher weight => smaller drift ===")
    for dw in (0.0, 0.5, 2.0, 10.0):
        ft, drift = run("full", dw)
        print(f"  drift_weight={dw:5.1f}: val_acc={ft.val_accuracy:.3f}  ||T-I||_F={drift:.3f}")

    print("\n=== bridge: SFT export_vectors -> GMS EmbeddingConfig Mode B ===")
    ft, _ = run("rotation", 1.0)
    names = ["overdraft_fee", "unfair", "escrow_shortage"]
    vecs = ft.export_vectors(names)                      # name-keyed transformed vectors
    p = Path("/tmp/sft_v_emb.pt")
    torch.save(vecs, p)

    from knowlytix.core.config import EmbeddingConfig
    from knowlytix.core.graph.embeddings import DualEmbedding
    from knowlytix.core.graph.encoders import init_dual_embeddings

    de = DualEmbedding(num_entities=len(names), d_v=D, d_u=D, m=128)
    init_dual_embeddings(de, names, EmbeddingConfig(
        v_vectors_path=str(p), u_vectors_path=str(p),
        warm_start=False, freeze_base=True))
    row0_ok = torch.allclose(de.v_embed.weight.data[0], vecs["overdraft_fee"], atol=1e-5)
    print(f"  exported {len(names)} vecs (dim={vecs['unfair'].shape[0]}) -> loaded into GMS "
          f"DualEmbedding: row0==export={row0_ok}  v_frozen={not de.v_embed.weight.requires_grad}")
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
