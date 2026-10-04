from dataclasses import dataclass
from typing import Any

import mteb

from src.config import logger


@dataclass(frozen=True)
class RetrievalDataset:
    task_name: str
    doc_texts: list[str]
    doc_ids: list[str]
    query_texts: list[str]
    query_ids: list[str]
    qrels: dict[str, dict[str, int]]
    instruction: str


def load_retrieval_dataset(task_name: str) -> RetrievalDataset:
    """Load an MTEB v2 retrieval benchmark (FiQA2018, ArguAna, SCIDOCS, TRECCOVID).

    All 4 retrieval tasks share an identical structure in MTEB v2:
        - Split container: task.dataset["default"]["test"]
        - Sub-keys: 'corpus', 'queries', 'relevant_docs'
        - Corpus columns: 'id' (str), 'title' (str), 'text' (str)
        - Queries columns: 'id' (str), 'text' (str)
        - Relevant docs (qrels): dict[str, dict[str, int]]
        - Task instruction: task.metadata.prompt["query"]
    """
    logger.info("Loading retrieval task: %s", task_name)
    task: Any = mteb.get_task(task_name)
    task.load_data()

    split = task.dataset["default"]["test"]
    corpus = split["corpus"]
    queries = split["queries"]
    qrels = split["relevant_docs"]

    # Prepend non-empty title to body text with a space according to standard BEIR protocol
    doc_texts = [
        f"{t} {x}".strip() if t else x
        for t, x in zip(corpus["title"], corpus["text"])
    ]

    instruction = task.metadata.prompt.get("query", "") if task.metadata and task.metadata.prompt else ""

    logger.info(
        "Loaded %s: %d documents, %d queries, %d qrel entries",
        task_name,
        len(doc_texts),
        len(queries["text"]),
        len(qrels),
    )

    return RetrievalDataset(
        task_name=task_name,
        doc_texts=doc_texts,
        doc_ids=list(corpus["id"]),
        query_texts=list(queries["text"]),
        query_ids=list(queries["id"]),
        qrels=qrels,
        instruction=instruction,
    )
