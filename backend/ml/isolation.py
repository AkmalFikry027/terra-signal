"""Multi-indicator Isolation Forest: V = [Z_temp, dNDVI%, Z_rain, dFire%] per grid cell."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config

FEATURES = ["z_temp", "d_ndvi_pct", "z_rain", "d_fire_pct"]
_LABELS = {
    "z_temp": ("warmer than baseline", "cooler than baseline"),
    "d_ndvi_pct": ("greener than baseline", "vegetation decline"),
    "z_rain": ("wetter than normal", "rainfall deficit"),
    "d_fire_pct": ("high local fire activity", "little fire activity"),
}


def score_cells(matrix: np.ndarray, feature_labels: list[tuple[str, str]] | None = None) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Returns (is_outlier bool[], anomaly_score 0..1 [], drivers per cell)."""
    n = len(matrix)
    if n < config.IFOREST_MIN_CELLS:
        return np.zeros(n, bool), np.zeros(n), [[] for _ in range(n)]
    scaled = StandardScaler().fit_transform(matrix)
    model = IsolationForest(n_estimators=200, contamination=config.IFOREST_CONTAMINATION, random_state=42)
    labels = model.fit_predict(scaled)
    raw = -model.score_samples(scaled)                       # higher = more anomalous
    span = raw.max() - raw.min()
    score = (raw - raw.min()) / span if span else np.zeros(n)
    drivers = []
    features = feature_labels or [_LABELS[key] for key in FEATURES[:matrix.shape[1]]]
    if len(features) != matrix.shape[1]:
        raise ValueError("feature_labels must match the input matrix columns")
    for i in range(n):
        order = np.argsort(-np.abs(scaled[i]))[:2]
        drivers.append([features[j][0 if scaled[i, j] >= 0 else 1] for j in order
                        if abs(scaled[i, j]) >= 1.0])
    return labels == -1, score, drivers
