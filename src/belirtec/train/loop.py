from __future__ import annotations

import copy
import os

from belirtec.train.train_config import TrainingConfig, _build, load_training_config


# --------------------------------------------------------------------------- #
# stability guards                                                             #
# --------------------------------------------------------------------------- #
def _nonfinite_grad_guard():
    """Trainer callback that skips optimizer steps whose gradients are non-finite.

    bf16 autocast has no GradScaler, so nothing natively detects inf/NaN
    gradients; one poisoned batch flows through ``clip_grad_norm_`` (which turns
    an inf norm into NaN scaling) straight into permanently-NaN weights.
    Zeroing all grads makes ``optimizer.step()`` a no-op for that batch, which
    is exactly what fp16's GradScaler does automatically.
    """
    import torch
    from transformers import TrainerCallback

    class NonFiniteGradSkip(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if model is None:
                return
            for p in model.parameters():
                g = p.grad
                if g is not None and not torch.isfinite(g).all():
                    model.zero_grad(set_to_none=True)
                    print(
                        f"[grad-guard] non-finite gradient at step "
                        f"{state.global_step}; update skipped"
                    )
                    return

    return NonFiniteGradSkip()


def _merge_lora(model) -> bool:
    """Merge trained LoRA weights into the base transformer, in place.

    After merging, the model is a plain transformer again and
    ``save_pretrained`` writes a standalone checkpoint (full weights, no
    adapter files, loads without peft).

    Handles both construction styles:
      * PeftModel wrap (our ``attach_lora``): ``merge_and_unload()``.
      * In-place injected adapters (transformers' ``add_adapter``): merge each
        LoRA layer's weights into its base module, swap the base module back
        in, and clear transformers' peft bookkeeping so ``save_pretrained``
        writes full weights instead of adapter-only.

    Returns True if a merge happened, False if there was nothing to merge.
    ``safe_merge=True`` makes peft verify the merged weights are finite.
    """
    auto_model = model[0].auto_model

    if hasattr(auto_model, "merge_and_unload"):
        model[0].auto_model = auto_model.merge_and_unload(safe_merge=True)
        return True

    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        return False

    lora_layers = [
        (name, mod) for name, mod in auto_model.named_modules()
        if isinstance(mod, BaseTunerLayer)
    ]
    if not lora_layers:
        return False

    for name, mod in lora_layers:
        mod.merge(safe_merge=True)
        parent = auto_model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else auto_model
        setattr(parent, name.rsplit(".", 1)[-1], mod.get_base_layer())

    if getattr(auto_model, "_hf_peft_config_loaded", False):
        auto_model._hf_peft_config_loaded = False
    if hasattr(auto_model, "peft_config"):
        try:
            del auto_model.peft_config
        except Exception:
            pass
    return True


# --------------------------------------------------------------------------- #
# training                                                                     #
# --------------------------------------------------------------------------- #
def _training_args(cfg: TrainingConfig, output_dir: str, has_eval: bool):
    from sentence_transformers import SentenceTransformerTrainingArguments

    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        learning_rate=cfg.train.lr,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        # torch>=2.8 silently flips the HF default optimizer to adamw_torch_fused,
        # whose fused kernels have a NaN history (pytorch/pytorch#95781). Pin the
        # unfused optimizer the champion recipe was validated on.
        optim="adamw_torch",
        # The default (True) replaces NaN/Inf losses with recent averages when
        # logging, which hides divergence. Log the truth.
        logging_nan_inf_filter=False,
        fp16=(cfg.train.precision == "fp16"),
        bf16=(cfg.train.precision == "bf16"),
        logging_steps=cfg.train.logging_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=None,
        eval_strategy=("steps" if has_eval else "no"),
        eval_steps=cfg.train.eval_steps,
        load_best_model_at_end=False,
        dataloader_drop_last=True,
        report_to="none",
        seed=cfg.train.seed,
    )
    if cfg.train.no_duplicates:
        try:
            from sentence_transformers.training_args import BatchSamplers

            if hasattr(BatchSamplers, "NO_DUPLICATES"):
                kwargs["batch_sampler"] = BatchSamplers.NO_DUPLICATES
                print("[ok] BatchSamplers.NO_DUPLICATES")
        except Exception as e:
            print(f"[warn] NO_DUPLICATES unavailable ({e})")
    return SentenceTransformerTrainingArguments(**kwargs)


def _run_one(cfg: TrainingConfig, output_dir: str, init_from: str | None):
    from sentence_transformers import SentenceTransformerTrainer
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

    from belirtec.train.data import build_datasets
    from belirtec.train.loss import build_losses
    from belirtec.train.model import attach_lora, build_model

    os.makedirs(output_dir, exist_ok=True)

    if init_from:
        # Staged runs chain checkpoints. Each phase's final model is MERGED
        # (standalone), so a phase that trains with LoRA attaches a FRESH
        # adapter on top of the previous phase's merged weights.
        from sentence_transformers import SentenceTransformer

        print(f"[phase] init from {init_from}")
        model = SentenceTransformer(init_from)
        model.max_seq_length = cfg.model.max_seq_length
        if cfg.model.lora.enabled:
            attach_lora(model, cfg)
    else:
        model = build_model(cfg)

    train_datasets, counts, stsb_eval = build_datasets(cfg)
    print(f"[data] counts: {counts}")
    print(f"[data] datasets: { {k: len(v) for k, v in train_datasets.items()} }")

    losses = build_losses(model, cfg, list(train_datasets.keys()))

    evaluator = None
    if stsb_eval is not None:
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=stsb_eval[0], sentences2=stsb_eval[1], scores=stsb_eval[2],
            main_similarity=SimilarityFunction.COSINE, name="stsb-val",
        )

    args = _training_args(cfg, output_dir, has_eval=evaluator is not None)
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_datasets, loss=losses,
        evaluator=evaluator, callbacks=[_nonfinite_grad_guard()],
    )
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    if cfg.model.lora.enabled:
        try:
            if _merge_lora(model):
                print("[lora] adapter merged into base; saving standalone model")
            else:
                print("[lora] no adapter layers found; saving as-is")
        except Exception as e:
            print(f"[lora] merge failed ({e}); saving adapter form")
    model.save_pretrained(final_dir)
    print(f"[done] {final_dir}")
    return final_dir


def run(config_path: str | None = None, experiment: str | None = None):
    cfg = load_training_config(config_path, experiment)

    if not cfg.phases:
        # single-phase: the champion / baseline path
        return _run_one(cfg, cfg.output_dir, init_from=None)

    # staged: each phase overrides the base config; init_from chains checkpoints
    from belirtec.train.train_config import raw_merged

    base_raw = raw_merged(config_path, experiment)
    prev_final = None
    last = None
    checkpoints: dict[str, str] = {}
    for i, phase in enumerate(cfg.phases):
        name = phase.get("name", f"phase{i}")
        phase_raw = copy.deepcopy(base_raw)
        phase_raw.pop("phases", None)
        _deep_apply(phase_raw, phase)
        phase_cfg = _build(phase_raw)
        out = os.path.join(cfg.output_dir, name)

        init = None
        init_ref = phase.get("init_from")
        if init_ref == "PREV":
            init = prev_final
        elif init_ref in checkpoints:
            init = checkpoints[init_ref]
        print(f"\n{'=' * 60}\n[phase {i}] {name}  (init_from={init})\n{'=' * 60}")

        last = _run_one(phase_cfg, out, init_from=init)
        checkpoints[name] = last
        prev_final = last
    return last


def _deep_apply(base: dict, override: dict):
    for k, v in override.items():
        if k in ("name", "init_from"):
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_apply(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
