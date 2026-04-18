class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._keys = keys
        self.reset()

    def reset(self):
        self._data = {key: {"total": 0.0, "counts": 0, "average": 0.0} for key in self._keys}

    def update(self, key, value, n=1):
        if key not in self._data:
            self._data[key] = {"total": 0.0, "counts": 0, "average": 0.0}
        self._data[key]["total"] += float(value) * n
        self._data[key]["counts"] += n
        self._data[key]["average"] = self._data[key]["total"] / self._data[key]["counts"]

    def avg(self, key):
        return self._data.get(key, {}).get("average", 0.0)

    def result(self):
        return {key: self._data[key]["average"] for key in self._data}

    def keys(self):
        return list(self._data.keys())
