from __future__ import annotations

import copy
import os

from belirtec.train.train_config import TrainingConfig, _build, load_training_config


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
    from belirtec.train.model import build_model

    os.makedirs(output_dir, exist_ok=True)

    if init_from:
        # staged: continue from a prior phase checkpoint
        from sentence_transformers import SentenceTransformer

        print(f"[phase] init from {init_from}")
        model = SentenceTransformer(init_from)
        model.max_seq_length = cfg.model.max_seq_length
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
        model=model, args=args, train_dataset=train_datasets, loss=losses, evaluator=evaluator
    )
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
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
        print(f"\n{'='*60}\n[phase {i}] {name}  (init_from={init})\n{'='*60}")

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
