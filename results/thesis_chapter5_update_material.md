# Chapter 5 — Discussion update material

Use the actual values from the generated tables, not legacy values. The discussion should distinguish three different concepts:

- deterministic generation has no free-form stochastic text generation;
- CNN slot mistakes can still propagate into the deterministic report;
- VLM/RAG unsupported-content flags are automated diagnostics and are not clinical hallucination measurements;
- raw VLM output must be discussed separately from the final slot-grounded report delivered by the pipeline;
- the RAG candidate gate uses predicted slots only and must be described as preventing retrieval-induced degradation, not as proof that retrieval is clinically correct.

The key comparative interpretation should therefore be framed as a **reliability–expressiveness trade-off**, rather than as proof that the deterministic template is clinically safe.

The RAG branch is train-only retrieval with top-k=3. Retrieval is driven by CNN-predicted structured findings rather than a generated draft; retrieved reports are sanitised language/style context only and are explicitly not treated as evidence about the current patient.

Segmentation is an auxiliary spatial evaluation branch. Its predictions are **not passed into the VLM** in this experiment.
