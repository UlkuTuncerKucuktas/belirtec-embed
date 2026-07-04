from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class Lora:
    enabled: bool
    r: int
    alpha: int
    dropout: float
    targets: list[str]


@dataclass(frozen=True)
class ModelCfg:
    base: str
    max_seq_length: int
    lora: Lora


@dataclass(frozen=True)
class Forgetting:
    # mode: null | filter (keep safe: _forget_cos > threshold) |
    #       keep_risky (keep _forget_cos <= threshold) | weight (scale by margin)
    mode: str | None = None
    threshold: float = 0.0


@dataclass(frozen=True)
class BucketSpec:
    enabled: bool = True
    limit: int | None = None
    shuffle: bool = True
    seed: int | None = None
    forgetting: Forgetting = field(default_factory=Forgetting)
    model_filter: str | None = None


@dataclass(frozen=True)
class DataCfg:
    synthetic_source: str
    synthetic_repo: str
    local_paths: dict[str, str]
    real_nli_hf: str
    real_stsb_hf: str
    nli_samples: int
    num_negatives: int
    use_instructions: bool
    buckets: dict[str, BucketSpec]
    real_nli: BucketSpec
    msmarco: dict | None = None


@dataclass(frozen=True)
class LossCfg:
    scale: float
    matryoshka_dims: list[int]
    gist_guide: str
    per_dataset: dict[str, str]


@dataclass(frozen=True)
class TrainCfg:
    precision: str
    batch_size: int
    mini_batch_size: int
    epochs: float
    lr: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    no_duplicates: bool
    seed: int
    save_steps: int
    eval_steps: int
    logging_steps: int


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelCfg
    data: DataCfg
    loss: LossCfg
    train: TrainCfg
    phases: list[dict] | None
    output_dir: str


def _bucket_spec(d: dict | None) -> BucketSpec:
    d = dict(d or {})
    fg = Forgetting(**(d.pop("forgetting", {}) or {}))
    return BucketSpec(forgetting=fg, **d)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build(raw: dict) -> TrainingConfig:
    m = raw["model"]
    model = ModelCfg(base=m["base"], max_seq_length=m["max_seq_length"], lora=Lora(**m["lora"]))

    d = raw["data"]
    data = DataCfg(
        synthetic_source=d["synthetic_source"],
        synthetic_repo=d["synthetic_repo"],
        local_paths=dict(d["local_paths"]),
        real_nli_hf=d["real_nli_hf"],
        real_stsb_hf=d["real_stsb_hf"],
        nli_samples=d["nli_samples"],
        num_negatives=d["num_negatives"],
        use_instructions=d["use_instructions"],
        buckets={k: _bucket_spec(v) for k, v in d["buckets"].items()},
        real_nli=_bucket_spec(d.get("real_nli", {})),
        msmarco=d.get("msmarco"),
    )

    loss = LossCfg(**raw["loss"])
    train = TrainCfg(**raw["train"])
    return TrainingConfig(
        model=model, data=data, loss=loss, train=train,
        phases=raw.get("phases"), output_dir=raw["output_dir"],
    )


def load_training_config(path: str | Path | None = None, experiment: str | None = None) -> TrainingConfig:
    base_path = Path(path) if path else CONFIG_DIR / "training.yaml"
    with open(base_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if experiment:
        exp_path = CONFIG_DIR / "experiments" / f"{experiment}.yaml"
        with open(exp_path, encoding="utf-8") as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})
    return _build(raw)


def raw_merged(path: str | Path | None, experiment: str | None) -> dict:
    # phases need the raw per-phase override dicts (merged), not frozen dataclasses
    base_path = Path(path) if path else CONFIG_DIR / "training.yaml"
    with open(base_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if experiment:
        exp_path = CONFIG_DIR / "experiments" / f"{experiment}.yaml"
        with open(exp_path, encoding="utf-8") as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})
    return raw
