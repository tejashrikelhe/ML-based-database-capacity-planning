from __future__ import annotations
from sklearn.ensemble import GradientBoostingRegressor

def make_quantile_gbr(alpha: float, random_state: int = 42) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=350,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=random_state
    )
