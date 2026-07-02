from __future__ import annotations

import os

from belirtec.config import Model, Sampling, Vllm


def _vllm_env() -> None:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")


def _chat_template_kwargs(family: str) -> dict:
    # correctness, not tuning: Qwen must have thinking off or it emits <think> blocks that
    # break parse_json; gpt-oss uses the harmony reasoning_effort knob.
    if family == "qwen":
        return {"enable_thinking": False}
    if family == "gpt_oss":
        return {"reasoning_effort": "low"}
    return {}


def load(model: Model, vllm: Vllm, sampling: Sampling):
    _vllm_env()
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=model.id,
        tensor_parallel_size=1,
        gpu_memory_utilization=vllm.gpu_memory_utilization,
        max_model_len=vllm.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
    )
    ctk = _chat_template_kwargs(model.family)

    def gen(prompts, temp, max_tokens):
        sp = SamplingParams(
            temperature=temp, top_p=sampling.top_p, max_tokens=max_tokens, seed=sampling.seed
        )
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        kw = {"chat_template_kwargs": ctk} if ctk else {}
        outs = engine.chat(msgs, sp, **kw)
        return [o.outputs[0].text or "" for o in outs]

    return gen
