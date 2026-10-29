from __future__ import annotations

def convergence_time(event_time: int, observations: list[tuple[int, bool]]) -> int | None:
    '''
    Earliest time at which all remaining observations are correct.
    Returns None if stable convergence never occurs in the observation horizon.
    '''
    obs = sorted((t, ok) for t, ok in observations if t >= event_time)
    for i, (t, ok) in enumerate(obs):
        if ok and all(v for _, v in obs[i:]):
            return t - event_time
    return None

def exposure_window(event_time: int, exposures: list[int]) -> int:
    relevant = [t for t in exposures if t >= event_time]
    return 0 if not relevant else max(relevant) - event_time

def rate(flags: list[bool]) -> float:
    return 0.0 if not flags else sum(bool(x) for x in flags) / len(flags)
