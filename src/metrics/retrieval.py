import pytrec_eval


def compute_retrieval_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute nDCG@10, Recall@100, and MRR@10 in a single standard TREC evaluation pass.

    Mathematical Formulation:
        DCG@10 = sum_{i=1}^{10} (2^{rel_i} - 1) / log_2(i + 1), nDCG@10 = DCG@10 / IDCG@10
        Recall@100 = |{retrieved top 100} cap {relevant docs}| / |{relevant docs}|
        MRR@10 = (1 / |Q|) sum_{q in Q} (1 / rank_1)  if rank_1 <= 10 else 0

    Args:
        qrels: Ground-truth relevance mapping {query_id: {doc_id: relevance_score}}.
        run: Model retrieval run predictions {query_id: {doc_id: similarity_score}}.

    Returns:
        Dictionary containing 'ndcg_at_10', 'recall_at_100', and 'mrr_at_10'.
    """
    if not qrels:
        return {"ndcg_at_10": 0.0, "recall_at_100": 0.0, "mrr_at_10": 0.0}

    # Truncate run candidates to top-10 per query for MRR@10 evaluation
    run_top10 = {
        qid: dict(sorted(docs.items(), key=lambda item: item[1], reverse=True)[:10])
        for qid, docs in run.items()
    }

    # Evaluate nDCG@10 and Recall@100 on the full top-100 run in a single C pass
    evaluator_100 = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"})
    evaluator_10 = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})

    res_100 = evaluator_100.evaluate(run)
    res_10 = evaluator_10.evaluate(run_top10)

    # Standard TREC evaluation: average over all benchmark queries in qrels
    n_queries = len(qrels)
    ndcg = (
        sum(float(res_100[q]["ndcg_cut_10"]) if q in res_100 else 0.0 for q in qrels)
        / n_queries
    )
    recall = (
        sum(float(res_100[q]["recall_100"]) if q in res_100 else 0.0 for q in qrels)
        / n_queries
    )
    mrr = (
        sum(float(res_10[q]["recip_rank"]) if q in res_10 else 0.0 for q in qrels)
        / n_queries
    )

    return {"ndcg_at_10": ndcg, "recall_at_100": recall, "mrr_at_10": mrr}
