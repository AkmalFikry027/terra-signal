"""Statistical indicators and status classification (Z-score, percent change, percentile rules)."""
import config


def z_score(current: float, baseline: float, sigma: float) -> float:
    return (current - baseline) / sigma if sigma else 0.0


def pct_change(current: float, baseline: float) -> float:
    return 0.0 if baseline == 0 else (current - baseline) / abs(baseline) * 100.0


def temperature_status(z: float) -> str:
    a = abs(z)
    if a >= config.Z_EXTREME:
        return "EXTREME"
    if a >= config.Z_MODERATE:
        return "MODERATE"
    return "ELEVATED" if z >= 1 else "NORMAL"


def vegetation_status(change_pct: float) -> str:
    if change_pct <= config.NDVI_DECLINE_PCT:
        return "DECLINE"
    return "WATCH" if change_pct <= config.NDVI_WATCH_PCT else "HEALTHY"


def soil_wetness_status(change_pct: float) -> str:
    if change_pct <= config.SOIL_DRYING_PCT:
        return "DRYING"
    return "WATCH" if change_pct <= config.SOIL_WATCH_PCT else "NORMAL"


def precipitation_status(percentile: float) -> str:
    if percentile < config.RAIN_DEFICIT_PCTL:
        return "DEFICIT"
    return "EXTREME" if percentile >= config.RAIN_EXTREME_PCTL else "NORMAL"


def fire_status(change_pct: float, detections: int) -> str:
    if detections == 0:
        return "NORMAL"
    if change_pct >= config.FIRE_CRITICAL_PCT:
        return "CRITICAL"
    return "ELEVATED" if change_pct >= config.FIRE_ELEVATED_PCT else "NORMAL"
