from belirtec.train.data import _select, _apply_forgetting, _apply_limit, _pad_negs, _contrastive_row
from belirtec.train.train_config import BucketSpec, Forgetting, load_training_config


def _scored(coss):
    return [{"anchor": f"q{i}", "positive": f"p{i}", "_forget_cos": c} for i, c in enumerate(coss)]


def test_forgetting_filter_keeps_safe():
    rows = _scored([0.05, 0.02, -0.01, -0.03, 0.06])
    safe = _apply_forgetting(rows, BucketSpec(forgetting=Forgetting(mode="filter", threshold=0.0)))
    assert all(r["_forget_cos"] > 0 for r in safe)
    assert len(safe) == 3


def test_forgetting_keep_risky_keeps_tail():
    rows = _scored([0.05, 0.02, -0.01, -0.03, 0.06])
    risky = _apply_forgetting(rows, BucketSpec(forgetting=Forgetting(mode="keep_risky", threshold=0.0)))
    assert all(r["_forget_cos"] <= 0 for r in risky)
    assert len(risky) == 2


def test_forgetting_none_is_passthrough():
    rows = _scored([0.05, -0.01])
    assert len(_apply_forgetting(rows, BucketSpec(forgetting=Forgetting(mode=None)))) == 2


def test_forgetting_noop_on_unscored_bucket():
    rows = [{"anchor": "q", "positive": "p"}]  # no _forget_cos
    out = _apply_forgetting(rows, BucketSpec(forgetting=Forgetting(mode="filter")))
    assert len(out) == 1  # unscored -> no-op, not dropped


def test_limit_deterministic():
    rows = _scored([0.1, 0.2, 0.3, 0.4, 0.5])
    a = _apply_limit(rows, BucketSpec(limit=2, shuffle=True, seed=7), 42)
    b = _apply_limit(rows, BucketSpec(limit=2, shuffle=True, seed=7), 42)
    assert a == b and len(a) == 2


def test_limit_none_keeps_all():
    rows = _scored([0.1, 0.2, 0.3])
    assert len(_apply_limit(rows, BucketSpec(limit=None), 42)) == 3


def test_pad_negs():
    assert _pad_negs(["a", "b"], 4) == ["a", "b", "a", "b"]
    assert _pad_negs([], 4) is None
    assert _pad_negs(["a", "b", "c", "d", "e"], 4) == ["a", "b", "c", "d"]


def test_contrastive_row_rejects_degenerate():
    assert _contrastive_row("same", "same", ["n"], 4, None, False) is None
    assert _contrastive_row("x", "y", [], 4, None, False) is None  # no negs
    ok = _contrastive_row("question", "answer", ["neg"], 4, None, False)
    assert ok["anchor"] == "question" and ok["negative_1"] == "neg"


def test_instruction_formatting():
    r = _contrastive_row("soru metni", "p", ["n"], 4, "do the thing", True)
    assert r["anchor"].startswith("Instruct: do the thing")


def test_champion_config_defaults():
    c = load_training_config()
    assert c.model.base == "BAAI/bge-m3"
    assert c.model.lora.r == 32 and c.model.lora.alpha == 64
    assert c.train.lr == 2.0e-4 and c.train.no_duplicates is True
    assert c.loss.matryoshka_dims == [1024, 256]
    assert all(c.data.buckets[b].enabled for b in ["legal", "retrieval", "sts", "classification"])
