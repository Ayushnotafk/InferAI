import random

def format_confidence(confidence):
    return round(confidence, 2)

def get_random_confidence() -> float:
    """Return a random confidence value between 75.0 and 95.0."""
    return round(random.uniform(75.0, 95.0), 2)

# Backward-compatible alias (used by older deployed versions of app.py)
get_random_confid = get_random_confidence

