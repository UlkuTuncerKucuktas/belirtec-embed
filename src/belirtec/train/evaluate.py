from __future__ import annotations

# Thin wrapper over the vendored MTEB legal-evaluation harness (mteb_eval.py, kept
# unchanged from the original single-file script). This just calls its MTEBEvaluator
# so `scripts/eval.py` matches the repo's thin-CLI style.


def run(model_path: str, output_dir: str = "results", batch_size: int = 32,
        device: str | None = None, overwrite: bool = False):
    from belirtec.train.mteb_eval import MTEBEvaluator

    evaluator = MTEBEvaluator(
        model_name=model_path,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        overwrite_results=overwrite,
    )
    if not evaluator.load_model():
        raise RuntimeError(f"failed to load model: {model_path}")
    success, results, summary_df, summary_table_df = evaluator.run_evaluation()
    if not success:
        raise RuntimeError("evaluation failed or was cancelled")
    if summary_table_df is not None:
        print(summary_table_df.to_string(index=False))
    evaluator.cleanup()
    return summary_table_df
