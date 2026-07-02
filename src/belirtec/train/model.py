from __future__ import annotations

from belirtec.train.train_config import TrainingConfig


def build_model(cfg: TrainingConfig):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.model.base)
    model.max_seq_length = cfg.model.max_seq_length

    if cfg.model.lora.enabled:
        from peft import LoraConfig

        peft_cfg = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            lora_dropout=cfg.model.lora.dropout,
            bias="none",
            target_modules=list(cfg.model.lora.targets),
        )
        model.add_adapter(peft_cfg)
        print(f"[lora] adapters added (r={cfg.model.lora.r}, alpha={cfg.model.lora.alpha}); "
              f"base weights frozen")

    try:
        pooling = model[1].get_pooling_mode_str()
    except Exception:
        pooling = "?"
    print(f"[model] {cfg.model.base} | pooling={pooling} | max_seq={model.max_seq_length} | "
          f"lora={cfg.model.lora.enabled}")
    return model
