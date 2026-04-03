import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from .config import (
    SUPPORTED_SEQUENCE,
    SUPPORTED_TABULAR,
    SequenceConfig,
    TabularConfig,
    parse_bool_list,
    parse_int_list,
)
from .sequence import SequenceOptimizer
from .tabular import TabularOptimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified optimizer for tabular and sequence models")

    parser.add_argument("--family", type=str, choices=["tabular", "sequence", "all"], default="tabular")
    parser.add_argument("--data_folder", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="data/models/optimization")

    parser.add_argument("--mode", type=str, choices=["tune", "train", "all"], default="all")
    parser.add_argument("--model", type=str, choices=list(SUPPORTED_TABULAR) + ["all"], default="all")
    parser.add_argument("--train_percent", type=float, default=0.8)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--site", type=str, default="all")

    parser.add_argument("--sequence_model", type=str, choices=list(SUPPORTED_SEQUENCE) + ["all"], default="lstm")
    parser.add_argument("--seq_lengths", type=str, default="6,12,24,48")
    parser.add_argument("--drop_prod_options", type=str, default="false,true")
    parser.add_argument("--sequence_site_start", type=int, default=3)
    parser.add_argument("--sequence_site_end", type=int, default=9)
    parser.add_argument("--sequence_test_size", type=float, default=0.2)
    parser.add_argument("--sequence_no_cv", action="store_true")

    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    if args.family in {"tabular", "all"}:
        tabular_config = TabularConfig(
            model=args.model,
            site=args.site,
            train_percent=args.train_percent,
            n_trials=args.n_trials,
            mode=args.mode,
            data_folder=Path(args.data_folder),
            output_dir=output_dir,
        )
        results["tabular"] = TabularOptimizer(tabular_config).run()

    if args.family in {"sequence", "all"}:
        sequence_config = SequenceConfig(
            sequence_model=args.sequence_model,
            seq_lengths=parse_int_list(args.seq_lengths),
            drop_prod_options=parse_bool_list(args.drop_prod_options),
            site_start=args.sequence_site_start,
            site_end=args.sequence_site_end,
            test_size=args.sequence_test_size,
            no_cv=args.sequence_no_cv,
            output_dir=output_dir,
            model_root=Path("data/models"),
        )
        results["sequence"] = SequenceOptimizer(sequence_config).run()

    summary_file = output_dir / "optimization_summary.json"
    summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    start = time.time()
    run(parse_args())
    logging.info("Completed in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
