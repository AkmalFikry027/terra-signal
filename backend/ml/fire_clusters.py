"""DBSCAN spatial clustering of FIRMS detections (haversine metric, no preset k)."""
import numpy as np
from sklearn.cluster import DBSCAN

import config
from ml.geo import EARTH_KM


def cluster_fires(detections: list[dict]) -> np.ndarray:
    """Return a label per detection: 0..n-1 for clusters, -1 for noise (isolated detections)."""
    if len(detections) < config.DBSCAN_MIN_SAMPLES:
        return np.full(len(detections), -1, dtype=int)
    coords = np.radians([[d["lat"], d["lon"]] for d in detections])
    return DBSCAN(eps=config.DBSCAN_EPS_KM / EARTH_KM, min_samples=config.DBSCAN_MIN_SAMPLES,
                  metric="haversine").fit_predict(coords)
