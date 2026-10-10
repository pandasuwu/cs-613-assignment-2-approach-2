import sys
from pathlib import Path

# Add project root to sys.path so 'src' is resolvable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

from src.config import ROOT_DIR, logger
from src.tables import (
    build_suite_1_retrieval,
    build_suite_2_similarity,
    build_suite_3_compression,
    build_suite_4_geometry,
    build_suite_5_ablation,
    build_suite_6_analysis,
    load_master_lazy,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile granular, publication-ready CSV research tables from atomic evaluation results."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        help="Specific table suite to generate (1..6 or all). Default: all.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT_DIR / "tables"),
        help="Directory to save generated CSV tables. Default: tables/.",
    )

    args = parser.parse_args()
    tables_dir = Path(args.out_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info(
        "Starting granular table generation pipeline (suite=%s, out_dir=%s)",
        args.suite,
        tables_dir,
    )

    master_lf = load_master_lazy()

    suites_to_run = (
        [args.suite] if args.suite != "all" else [str(i) for i in range(1, 7)]
    )

    for s in suites_to_run:
        logger.info("Executing Table Suite %s...", s)
        if s == "1":
            build_suite_1_retrieval(master_lf, tables_dir)
        elif s == "2":
            build_suite_2_similarity(master_lf, tables_dir)
        elif s == "3":
            build_suite_3_compression(master_lf, tables_dir)
        elif s == "4":
            build_suite_4_geometry(master_lf, tables_dir)
        elif s == "5":
            build_suite_5_ablation(master_lf, tables_dir)
        elif s == "6":
            build_suite_6_analysis(master_lf, tables_dir)
        else:
            logger.warning("Unknown table suite: %s (skipping)", s)

    logger.info(
        "Completed table generation pipeline in %.2f seconds.", time.time() - t0
    )


if __name__ == "__main__":
    main()
