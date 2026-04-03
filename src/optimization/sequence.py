import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import SequenceConfig, SequenceExperiment, resolve_sequence_models


@dataclass
class SequenceRunResult:
    row: dict[str, Any]
    failed: bool


def iter_sequence_experiments(config: SequenceConfig) -> Iterable[SequenceExperiment]:
    if config.site_start > config.site_end:
        raise ValueError("sequence_site_start must be <= sequence_site_end")

    models = resolve_sequence_models(config.sequence_model)

    for site_index in range(config.site_start, config.site_end + 1):
        for model_type in models:
            for seq_len in config.seq_lengths:
                for drop_prod in config.drop_prod_options:
                    yield SequenceExperiment(
                        site_index=site_index,
                        model_type=model_type,
                        seq_len=seq_len,
                        drop_prod=drop_prod,
                        test_size=config.test_size,
                        no_cv=config.no_cv,
                        model_root=config.model_root,
                    )


def _result_row_base(experiment: SequenceExperiment) -> dict[str, Any]:
    return {
        "site_index": experiment.site_index,
        "name": experiment.run_name,
        "model": experiment.model_type,
        "seq_len": experiment.seq_len,
        "drop_prod": experiment.drop_prod,
    }


def run_sequence_experiment(
    experiment: SequenceExperiment,
    run_pipeline: Callable[..., dict[str, Any]],
) -> SequenceRunResult:
    experiment.savepath.parent.mkdir(parents=True, exist_ok=True)
    row = _result_row_base(experiment)

    try:
        metrics = run_pipeline(
            model_type=experiment.model_type,
            seq_len=experiment.seq_len,
            drop_prod=experiment.drop_prod,
            no_cv=experiment.no_cv,
            savepath=str(experiment.savepath),
            test_size=experiment.test_size,
            one_site_only=True,
            idx_site=experiment.site_index,
        )

        row.update(
            {
                "mae": metrics.get("eval_mae"),
                "rmse": metrics.get("eval_rmse"),
                "nrmse_pct": (metrics.get("eval_nrmse") or 0.0) * 100,
                "portfolio_mae_total": metrics.get("portfolio_mae_total"),
                "portfolio_rmse_total": metrics.get("portfolio_rmse_total"),
                "portfolio_nrmse_total": metrics.get("portfolio_nrmse_total"),
                "portfolio_mae_per_site": metrics.get("portfolio_mae_per_site"),
                "portfolio_rmse_per_site": metrics.get("portfolio_rmse_per_site"),
                "error": "",
            }
        )
        return SequenceRunResult(row=row, failed=False)
    except Exception as exc:
        row.update(
            {
                "mae": None,
                "rmse": None,
                "nrmse_pct": None,
                "portfolio_mae_total": None,
                "portfolio_rmse_total": None,
                "portfolio_nrmse_total": None,
                "portfolio_mae_per_site": None,
                "portfolio_rmse_per_site": None,
                "error": str(exc),
            }
        )
        return SequenceRunResult(row=row, failed=True)


class SequenceOptimizer:
    def __init__(
        self,
        config: SequenceConfig,
        run_pipeline: Callable[..., dict[str, Any]] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._run_pipeline = run_pipeline or self._load_run_pipeline()

    def _load_run_pipeline(self) -> Callable[..., dict[str, Any]]:
        from main import run_pipeline

        return run_pipeline

    def run(self) -> list[dict[str, Any]]:
        output_dir = self.config.output_dir / "sequence_optimization"
        output_dir.mkdir(parents=True, exist_ok=True)

        rows_by_site: dict[int, list[dict[str, Any]]] = {}
        all_rows: list[dict[str, Any]] = []

        for experiment in iter_sequence_experiments(self.config):
            result = run_sequence_experiment(experiment, self._run_pipeline)
            rows_by_site.setdefault(experiment.site_index, []).append(result.row)
            all_rows.append(result.row)

            if result.failed:
                self.logger.warning("Sequence experiment failed: %s", experiment.run_name)

        for site_index, rows in rows_by_site.items():
            pd.DataFrame(rows).to_csv(output_dir / f"sequence_search_site_{site_index}.csv", index=False)

        if all_rows:
            pd.DataFrame(all_rows).to_csv(output_dir / "sequence_search_all_sites.csv", index=False)

        return all_rows
