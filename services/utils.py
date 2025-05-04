import numpy as np

def fix_array_string(s):
    try:
        return np.array([float(x) for x in s.strip("[]").split()])
    except Exception:
        return None
