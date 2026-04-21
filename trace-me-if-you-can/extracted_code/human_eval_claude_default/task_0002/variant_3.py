return float(str(number).split('.')[-1]) / (10 ** len(str(number).split('.')[-1])) if '.' in str(number) else 0.0
