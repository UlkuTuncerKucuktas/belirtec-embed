import json

from belirtec.config import load_config
from belirtec.grounded import generate
from belirtec.io import Passage


def test_schema_and_provenance():
    cfg = load_config()

    def stub(prompts, temp, max_tokens):
        return [json.dumps({"query": "ornek soru budur nedir"}) for _ in prompts]

    rows = list(generate(stub, [Passage("id1", "metin")], cfg.axes["legal"], "hukuki", cfg.sampling))
    assert len(rows) == 1
    assert set(rows[0]) == {"anchor", "positive", "source", "persona", "intent", "difficulty"}
    assert rows[0]["source"] == "id1"


def test_short_query_dropped():
    cfg = load_config()

    def stub(prompts, temp, max_tokens):
        return [json.dumps({"query": "kısa"}) for _ in prompts]

    rows = list(generate(stub, [Passage("id1", "metin")], cfg.axes["legal"], "hukuki", cfg.sampling))
    assert rows == []
