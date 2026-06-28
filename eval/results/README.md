# Evaluation results

Files in this directory are reproducible development outputs.

`official_retrieval_comparison_calibrated.json` compares BM25, dense retrieval,
and hybrid reciprocal-rank fusion on the small official-source development
benchmark.

The dense cosine-similarity threshold of 0.67 is a development calibration. It
must be recalibrated on a separate calibration split before evaluation on an
independent final test set. These development results are not final publication
evidence.
