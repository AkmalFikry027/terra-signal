import numpy as np

EARTH_KM = 6371.0088


def haversine_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Pairwise great-circle distance in km, shape (len(1), len(2))."""
    la1, lo1 = np.radians(np.asarray(lat1, float))[:, None], np.radians(np.asarray(lon1, float))[:, None]
    la2, lo2 = np.radians(np.asarray(lat2, float))[None, :], np.radians(np.asarray(lon2, float))[None, :]
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))
