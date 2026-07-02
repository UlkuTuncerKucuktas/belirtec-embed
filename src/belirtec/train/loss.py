from __future__ import annotations

from belirtec.train.train_config import TrainingConfig

# guide models are expensive to load; cache one per guide id across datasets
_GUIDE_CACHE: dict = {}


def _guide(guide_id: str):
    if guide_id not in _GUIDE_CACHE:
        from sentence_transformers import SentenceTransformer

        _GUIDE_CACHE[guide_id] = SentenceTransformer(guide_id)
    return _GUIDE_CACHE[guide_id]


def _base_loss(kind: str, model, cfg: TrainingConfig):
    from sentence_transformers.losses import (
        CachedGISTEmbedLoss,
        CachedMultipleNegativesRankingLoss,
        CoSENTLoss,
    )

    if kind == "mnrl":
        return CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=cfg.train.mini_batch_size, scale=cfg.loss.scale
        )
    if kind == "gist":
        # NewMind recipe: guide model filters in-batch false negatives.
        return CachedGISTEmbedLoss(
            model, guide=_guide(cfg.loss.gist_guide), mini_batch_size=cfg.train.mini_batch_size
        )
    if kind == "cosent":
        return CoSENTLoss(model)
    raise ValueError(f"unknown loss kind: {kind}")


def build_losses(model, cfg: TrainingConfig, dataset_names: list[str]) -> dict:
    """One Matryoshka-wrapped loss per training dataset, per configs' per_dataset map."""
    from sentence_transformers.losses import MatryoshkaLoss

    losses = {}
    for name in dataset_names:
        kind = cfg.loss.per_dataset.get(name)
        if kind is None:
            raise ValueError(f"no loss configured for dataset '{name}' in loss.per_dataset")
        base = _base_loss(kind, model, cfg)
        losses[name] = MatryoshkaLoss(model, base, matryoshka_dims=list(cfg.loss.matryoshka_dims))
        print(f"[loss] {name}: {kind} + Matryoshka{cfg.loss.matryoshka_dims}")
    return losses
