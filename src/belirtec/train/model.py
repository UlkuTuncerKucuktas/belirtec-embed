from __future__ import annotations

from belirtec.train.train_config import TrainingConfig


def attach_lora(model, cfg: TrainingConfig):
    """Attach a LoRA adapter to the SentenceTransformer's inner transformer.

    Uses peft's ``get_peft_model`` (PeftModel wrap) rather than transformers'
    in-place ``add_adapter`` injection: the PeftModel wrap is what provides
    ``merge_and_unload()``, so trained adapters can be merged into the base
    weights for standalone checkpoints. Training behaviour is identical either
    way — base weights frozen, only adapter parameters receive gradients.
    """
    from peft import LoraConfig, get_peft_model

    peft_cfg = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
        bias="none",
        target_modules=list(cfg.model.lora.targets),
    )
    model[0].auto_model = get_peft_model(model[0].auto_model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[lora] PeftModel wrap (r={cfg.model.lora.r}, alpha={cfg.model.lora.alpha}); "
        f"trainable {trainable:,}/{total:,} ({100 * trainable / total:.2f}%)"
    )
    return model


def build_model(cfg: TrainingConfig):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.model.base)
    model.max_seq_length = cfg.model.max_seq_length

    if cfg.model.lora.enabled:
        attach_lora(model, cfg)

    try:
        pooling = model[1].get_pooling_mode_str()
    except Exception:
        pooling = "?"
    print(
        f"[model] {cfg.model.base} | pooling={pooling} | max_seq={model.max_seq_length} | "
        f"lora={cfg.model.lora.enabled}"
    )
    return model
