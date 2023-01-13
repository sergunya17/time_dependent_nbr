import argparse
import os

from src.dataset import DATASETS
from src.models import MODELS
from src.metrics import METRICS
from src.settings import DATA_DIR
from src.evaluation import Evaluator
from src.hypertuning import run_search


def run_experiment(
    dataset: str,
    model: str,
    metric: str,
    cutoff: int,
    num_trials: int,
    batch_size: int,
    dataset_dir_name: str | None,
    verbose=True,
):
    cutoff_list = [5, 10, 15, 20, 50, 100]

    if dataset in DATASETS.keys():
        dataset_cls = DATASETS[dataset]
    else:
        raise ValueError(f"Dataset {dataset} is unknown")

    if model in MODELS.keys():
        model_cls = MODELS[model]
    else:
        raise ValueError(f"Model {model} is unknown")

    if metric not in METRICS.keys():
        raise ValueError(f"Metric {metric} is unknown")

    if cutoff <= 0 or cutoff not in cutoff_list:
        raise ValueError(f"Invalid cutoff: {cutoff}")

    if num_trials < 1:
        raise ValueError(f"Invalid num_trials: {num_trials}")

    if batch_size < 1:
        raise ValueError(f"Invalid batch_size: {batch_size}")

    if dataset_dir_name is None:
        dataset_dir_name = dataset
    else:
        dataset_full_path = os.path.join(DATA_DIR, dataset_dir_name)
        if not os.path.exists(dataset_full_path):
            raise ValueError(f"Dataset path doesn't exist: {dataset_full_path}")

    data = dataset_cls(dataset_dir_name, verbose=verbose)
    data.load_split()

    evaluator_valid = Evaluator(dataset_df=data.val_df, cutoff_list=cutoff_list, batch_size=batch_size, verbose=verbose)
    evaluator_test = Evaluator(dataset_df=data.test_df, cutoff_list=cutoff_list, batch_size=batch_size, verbose=verbose)

    exp_prefix = f"{dataset_dir_name}_{model}"
    run_search(
        dataset=data,
        model_cls=model_cls,
        evaluator_valid=evaluator_valid,
        evaluator_test=evaluator_test,
        metric=metric,
        cutoff=cutoff,
        num_trials=num_trials,
        prefix=exp_prefix,
    )


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--metric", type=str, default="recall")
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dataset-dir-name", type=str, default=None)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args, _ = parser.parse_known_args()
    run_experiment(
        dataset=args.dataset,
        model=args.model,
        metric=args.metric,
        cutoff=args.cutoff,
        num_trials=args.num_trials,
        batch_size=args.batch_size,
        dataset_dir_name=args.dataset_dir_name,
    )
