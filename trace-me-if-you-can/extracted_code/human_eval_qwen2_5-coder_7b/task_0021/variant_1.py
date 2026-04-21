def normalize_values(values):
        min_val = min(values)
        max_val = max(values)
        return [(val - min_val) / (max_val - min_val) for val in values]
