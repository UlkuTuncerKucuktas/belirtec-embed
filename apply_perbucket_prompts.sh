#!/usr/bin/env bash
# Complete prompt-injection patches for gemma/qwen training.
# Run from repo root. Idempotent-ish (asserts will fail if already applied).
set -e
cd "$(dirname "$0")"

python - <<'PYEOF'
# ============ PATCH A: train_config.py — DataCfg fields + loader passthrough ============
p = "src/belirtec/train/train_config.py"
s = open(p).read()

# A1: add fields to DataCfg. Place BEFORE msmarco (the last field), WITH defaults
#     so field-ordering is valid (defaults after non-defaults) and existing
#     configs without these keys still work.
old_fields = '''    buckets: dict[str, BucketSpec]
    real_nli: BucketSpec
    msmarco: dict | None = None'''
new_fields = '''    buckets: dict[str, BucketSpec]
    real_nli: BucketSpec
    query_prompt: str | None = None
    document_prompt: str | None = None
    msmarco: dict | None = None'''
assert old_fields in s, "DataCfg fields block not found (already patched?)"
s = s.replace(old_fields, new_fields)

# A2: loader passthrough — read from yaml data block with .get (backward compatible)
old_build = '''        real_nli=_bucket_spec(d.get("real_nli", {})),
        msmarco=d.get("msmarco"),
    )'''
new_build = '''        real_nli=_bucket_spec(d.get("real_nli", {})),
        query_prompt=d.get("query_prompt"),
        document_prompt=d.get("document_prompt"),
        msmarco=d.get("msmarco"),
    )'''
assert old_build in s, "DataCfg construction not found"
s = s.replace(old_build, new_build)

open(p, "w").write(s)
print("PATCH A (train_config.py): DataCfg fields + loader passthrough  OK")


# ============ PATCH B: data.py — remaining 2 call sites (176, 205) ============
p2 = "src/belirtec/train/data.py"
s2 = open(p2).read()

# B1: line 176 (real_nli) — single-line call
old_nli = '''        row = _contrastive_row(ex[a_c], ex[p_c], [ex[n_c]], data.num_negatives, None, data.use_instructions)'''
new_nli = '''        row = _contrastive_row(ex[a_c], ex[p_c], [ex[n_c]], data.num_negatives, None,
                               data.use_instructions, data.query_prompt, data.document_prompt)'''
assert old_nli in s2, "real_nli call site not found"
s2 = s2.replace(old_nli, new_nli)

# B2: line 205 (msmarco) — multi-line call
old_ms = '''        row = _contrastive_row(
            ex.get("query_text"), ex.get("pos_text"), [ex.get("neg_text")],
            data.num_negatives, None, data.use_instructions,
        )'''
new_ms = '''        row = _contrastive_row(
            ex.get("query_text"), ex.get("pos_text"), [ex.get("neg_text")],
            data.num_negatives, None, data.use_instructions,
            data.query_prompt, data.document_prompt,
        )'''
assert old_ms in s2, "msmarco call site not found"
s2 = s2.replace(old_ms, new_ms)

open(p2, "w").write(s2)
print("PATCH B (data.py): real_nli + msmarco call sites  OK")
PYEOF

echo ""
echo "=== verify: all 4 _contrastive_row sites now consistent ==="
grep -n "query_prompt\|document_prompt" src/belirtec/train/data.py | head
echo ""
echo "=== import check ==="
python -c "import sys; sys.path.insert(0,'src'); from belirtec.train.train_config import DataCfg; from belirtec.train import data; print('imports OK; DataCfg has query_prompt:', 'query_prompt' in DataCfg.__dataclass_fields__)"
