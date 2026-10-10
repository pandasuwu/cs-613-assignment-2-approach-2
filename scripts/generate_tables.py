import sys
from pathlib import Path

# Add project root to sys.path so 'src' is resolvable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

from src.config import ROOT_DIR, logger
from src.tables import (
    build_table_1_retrieval,
    build_table_2_similarity,
    build_table_3_ablation,
    build_table_4_gap_closure,
    build_table_5_compression,
    build_table_6_parameters,
    build_table_7_geometry,
    build_table_8_correlation,
    build_table_9_context,
    load_master_lazy,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile publication-ready CSV research tables from atomic evaluation results."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        help="Specific table suite to generate (1..9 or all). Default: all.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT_DIR / "tables"),
        help="Directory to save generated CSV tables. Default: tables/.",
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info(
        "Starting table generation pipeline (suite=%s, out_dir=%s)", args.suite, out_dir
    )

    master_lf = load_master_lazy()

    suites_to_run = (
        [args.suite] if args.suite != "all" else [str(i) for i in range(1, 10)]
    )

    for s in suites_to_run:
        logger.info("Executing Table Suite %s...", s)
        if s == "1":
            build_table_1_retrieval(master_lf, out_dir)
        elif s == "2":
            build_table_2_similarity(master_lf, out_dir)
        elif s == "3":
            build_table_3_ablation(master_lf, out_dir)
        elif s == "4":
            build_table_4_gap_closure(master_lf, out_dir)
        elif s == "5":
            build_table_5_compression(master_lf, out_dir)
        elif s == "6":
            build_table_6_parameters(master_lf, out_dir)
        elif s == "7":
            build_table_7_geometry(master_lf, out_dir)
        elif s == "8":
            build_table_8_correlation(master_lf, out_dir)
        elif s == "9":
            build_table_9_context(master_lf, out_dir)
        else:
            logger.warning("Unknown table suite: %s (skipping)", s)

    logger.info(
        "Completed table generation pipeline in %.2f seconds.", time.time() - t0
    )


if __name__ == "__main__":
    main()
