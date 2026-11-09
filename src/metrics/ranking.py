#  Copyright 2026 David Población Criado
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
from typing import Dict, Tuple

import pandas as pd


def compute_recall_at_k(
        df: pd.DataFrame,
        predictors: Dict[str, bool],
        ground_truth: str = "acc_drop",
        fractions: Tuple[float, ...] = (0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Fraction of the true top-K most sensitive layers each predictor recovers."""
    if ground_truth not in df.columns:
        raise KeyError(f"Ground-truth column '{ground_truth}' not in the table")

    missing = [name for name in predictors if name not in df.columns]
    if missing:
        logging.warning("Predictors absent from the table, skipped: %s", missing)

    population = df.dropna(subset=[ground_truth])
    if len(population) < len(df):
        logging.warning(
            "Dropped %d of %d layers with no ground truth.",
            len(df) - len(population),
            len(df),
        )
    if population.empty:
        raise ValueError(f"No layer has a '{ground_truth}' value")

    n_layers = len(population)
    ks = {fraction: max(1, int(n_layers * fraction)) for fraction in fractions}
    true_top = {
        fraction: set(population.nlargest(k, ground_truth).index)
        for fraction, k in ks.items()
    }

    rows = []
    for name, higher_is_more_sensitive in predictors.items():
        if name not in population.columns:
            continue
        n_nan = int(population[name].isna().sum())
        if n_nan:
            logging.warning(
                "Predictor '%s' is NaN in %d of %d layers; those can never be "
                "in its top-K.",
                name,
                n_nan,
                n_layers,
            )

        row: Dict[str, object] = {"predictor": name, "layers": n_layers}
        for fraction, k in ks.items():
            if higher_is_more_sensitive:
                pred_top_k = set(population.nlargest(k, name).index)
            else:
                pred_top_k = set(population.nsmallest(k, name).index)
            row[f"recall@{int(fraction * 100)}%"] = round(
                len(true_top[fraction] & pred_top_k) / k, 3
            )
        rows.append(row)

    if not rows:
        raise ValueError("No usable predictor column found in the table")

    df_recall = pd.DataFrame(rows).set_index("predictor")
    return df_recall.sort_values(
        by=f"recall@{int(fractions[0] * 100)}%", ascending=False
    )
