"""
Combines the per-axis shard CSVs produced by `compute_llm_judge.py --axis ...`
(one shard per axis, each holding the full row set with only that axis's
judge columns filled in) into the single "<dataset>_with_judge.csv" the rest
of the pipeline expects.

Usage:
    python3 merge_judge_shards.py --dataset iclinicqa
"""

import argparse
import os

import pandas as pd

from config_llm_judge import AXES

HERE = os.path.dirname(__file__)


def load_config(dataset):
    if dataset == "medquad":
        import config_medquad as cfg
    elif dataset == "iclinicqa":
        import config_iclinicqa as cfg
    elif dataset == "medquad_ext":
        import config_medquad_ext as cfg
    elif dataset == "medredqa_ext":
        import config_medredqa_ext as cfg
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["medquad", "iclinicqa", "medquad_ext", "medredqa_ext"])
    args = parser.parse_args()

    cfg = load_config(args.dataset)
    df = pd.read_csv(cfg.DATA_PATH)

    for axis_key in AXES:
        shard_path = os.path.join(HERE, f"{args.dataset}_with_judge_{axis_key}.csv")
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"missing shard for axis '{axis_key}': {shard_path}")
        shard = pd.read_csv(shard_path)
        if len(shard) != len(df):
            raise ValueError(f"{shard_path} has {len(shard)} rows, expected {len(df)}")
        new_cols = [c for c in shard.columns if c not in df.columns]
        for col in new_cols:
            df[col] = shard[col]
        print(f"[{axis_key}] merged {len(new_cols)} columns from {shard_path}")

    out_path = os.path.join(HERE, f"{args.dataset}_with_judge.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
