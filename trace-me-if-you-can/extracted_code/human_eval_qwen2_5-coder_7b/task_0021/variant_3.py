def scale_data(data):
        lower_bound = min(data)
        upper_bound = max(data)
        return [(d - lower_bound) / (upper_bound - lower_bound) for d in data]
