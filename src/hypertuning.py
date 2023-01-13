from typing import Dict, Type
from functools import partial
import os

import optuna
import pandas as pd

from src.evaluation import Evaluator
from src.settings import RESULTS_DIR
from src.utils import set_global_seed
from src.models import IRecommender
from src.dataset import NBRDatasetBase


def _objective(
    trial: optuna.Trial,
    dataset: NBRDatasetBase,
    model_cls: Type[IRecommender],
    evaluator: Evaluator,
    metric: str,
    cutoff: int,
):
    set_global_seed(42)
    fit_params = model_cls.sample_params(trial)
    model = model_cls(**fit_params)

    set_global_seed(42)
    model.fit(dataset=dataset)

    set_global_seed(42)
    performance = None
    performance_dict = evaluator.evaluate_recommender(model)
    for metric_name, metric_value in performance_dict.items():
        if metric_name == f"{metric}@{cutoff:03d}":
            performance = metric_value
        trial.set_user_attr(metric_name, metric_value)

    return performance


def _run_optuna(
    dataset: NBRDatasetBase,
    model_cls: Type[IRecommender],
    evaluator: Evaluator,
    metric: str,
    cutoff: int,
    study_name: str,
    n_trials: int,
    hypertuning_db: str,
):

    set_global_seed(42)
    # We choose direction='maximize' to maximize returns as a goal.
    study = optuna.create_study(
        storage=hypertuning_db,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True,
    )

    set_global_seed(42)
    # Optimize for given study.
    study.optimize(
        partial(
            _objective,
            model_cls=model_cls,
            dataset=dataset,
            evaluator=evaluator,
            metric=metric,
            cutoff=cutoff,
        ),
        n_trials=n_trials,
    )

    return study.best_value


def _find_best_vparams(
    dataset: NBRDatasetBase,
    model_cls: Type[IRecommender],
    evaluator: Evaluator,
    metric: str,
    cutoff: int,
    num_trials: int,
    prefix: str,
) -> Dict:
    study_name = f"{prefix}"
    h_prefix = RESULTS_DIR.joinpath(f"{prefix}.db")
    hypertuning_db = f"sqlite:///{h_prefix}"

    set_global_seed(42)
    _run_optuna(
        model_cls=model_cls,
        dataset=dataset,
        evaluator=evaluator,
        metric=metric,
        cutoff=cutoff,
        study_name=study_name,
        hypertuning_db=hypertuning_db,
        n_trials=num_trials,
    )

    study = optuna.load_study(study_name=study_name, storage=hypertuning_db)
    df = study.trials_dataframe()
    logfile = RESULTS_DIR.joinpath(f"{prefix}_valid.csv")
    os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
    df.to_csv(logfile, index=False)

    best_params = study.best_params

    return best_params


def run_search(
    dataset: NBRDatasetBase,
    model_cls: Type[IRecommender],
    evaluator_valid: Evaluator,
    evaluator_test: Evaluator,
    metric: str,
    cutoff: int,
    num_trials: int,
    prefix: str,
):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    set_global_seed(42)
    best_vparams = _find_best_vparams(
        dataset=dataset,
        model_cls=model_cls,
        evaluator=evaluator_valid,
        metric=metric,
        cutoff=cutoff,
        num_trials=num_trials,
        prefix=prefix,
    )

    set_global_seed(42)
    best_vmodel = model_cls(**best_vparams)

    set_global_seed(42)
    best_vmodel.fit(dataset=dataset)

    set_global_seed(42)
    performance_dct = evaluator_test.evaluate_recommender(best_vmodel)

    df = pd.DataFrame([performance_dct])
    logfile = RESULTS_DIR.joinpath(f"{prefix}_test.csv")
    os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
    df.to_csv(logfile, index=False)
