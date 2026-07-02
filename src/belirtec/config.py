from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

_PLACEHOLDER = "<CONFIRM>"


@dataclass(frozen=True)
class Model:
    id: str
    family: str


@dataclass(frozen=True)
class Sampling:
    temp_tasks: float
    temp_samples: float
    top_p: float
    seed: int
    max_tokens_query: int
    max_tokens_samples: int


@dataclass(frozen=True)
class Vllm:
    max_model_len: int
    gpu_memory_utilization: float


@dataclass(frozen=True)
class LegalCorpus:
    clean_file: str
    raw_source: str
    eval_sets: list[str]
    text_field: str
    min_chars: int
    max_chars: int


@dataclass(frozen=True)
class RetrievalSource:
    hf_dataset: str
    hf_config: str | None
    text_field: str
    weight: float
    quality_filter: bool


@dataclass(frozen=True)
class RetrievalCorpus:
    sources: list[RetrievalSource]
    split: str
    streaming: bool
    min_chars: int
    max_chars: int


@dataclass(frozen=True)
class Corpora:
    legal: LegalCorpus
    retrieval: RetrievalCorpus


@dataclass(frozen=True)
class Axes:
    persona: list[str]
    intent: list[str]
    difficulty: dict[str, int]


@dataclass(frozen=True)
class Hub:
    dataset_repo: str


@dataclass(frozen=True)
class Config:
    models: list[Model]
    counts: dict[str, int]
    vllm: Vllm
    sampling: Sampling
    corpora: Corpora
    axes: dict[str, Axes]
    hub: Hub

    def model_ids(self) -> list[str]:
        return [m.id for m in self.models]


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file missing: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_dir: Path | None = None) -> Config:
    d = config_dir or CONFIG_DIR

    g = _read_yaml(d / "generation.yaml")
    models = [Model(**m) for m in g["models"]]
    bad = [m.id for m in models if _PLACEHOLDER in m.id or not m.id.strip()]
    if bad:
        raise ValueError(f"generation.yaml has unfilled model ids: {bad}")

    c = _read_yaml(d / "corpora.yaml")
    rc = c["retrieval"]
    sources = [RetrievalSource(**s) for s in rc["sources"]]
    _validate_weights(sources)
    retrieval = RetrievalCorpus(
        sources=sources,
        split=rc["split"],
        streaming=rc["streaming"],
        min_chars=rc["min_chars"],
        max_chars=rc["max_chars"],
    )
    corpora = Corpora(legal=LegalCorpus(**c["legal"]), retrieval=retrieval)

    a = _read_yaml(d / "axes.yaml")
    axes = {bucket: Axes(**spec) for bucket, spec in a.items()}

    return Config(
        models=models,
        counts=dict(g["counts"]),
        vllm=Vllm(**g["vllm"]),
        sampling=Sampling(**g["sampling"]),
        corpora=corpora,
        axes=axes,
        hub=Hub(**g["hub"]),
    )


def _validate_weights(sources: list[RetrievalSource]) -> None:
    if not sources:
        raise ValueError("retrieval.sources is empty")
    total = sum(s.weight for s in sources)
    if total <= 0:
        raise ValueError("retrieval source weights sum to <= 0")
