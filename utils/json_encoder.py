import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types"""

    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
