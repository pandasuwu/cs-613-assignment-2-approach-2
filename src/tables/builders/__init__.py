from src.tables.builders.suite_1_retrieval import build_suite_1_retrieval
from src.tables.builders.suite_2_similarity import build_suite_2_similarity
from src.tables.builders.suite_3_compression import build_suite_3_compression
from src.tables.builders.suite_4_geometry import build_suite_4_geometry
from src.tables.builders.suite_5_ablation import build_suite_5_ablation
from src.tables.builders.suite_6_analysis import build_suite_6_analysis

__all__ = [
    "build_suite_1_retrieval",
    "build_suite_2_similarity",
    "build_suite_3_compression",
    "build_suite_4_geometry",
    "build_suite_5_ablation",
    "build_suite_6_analysis",
]
