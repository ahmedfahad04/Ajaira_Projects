class ClassificationMetrics:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def compute(self, pred, actual):
        if pred == actual:
            if pred == 1:
                self.tp += 1
            else:
                self.tn += 1
        else:
            if pred == 1:
                self.fp += 1
            else:
                self.fn += 1

    def precision_calc(self, pred, actual):
        self.compute(pred, actual)
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    def recall_calc(self, pred, actual):
        self.compute(pred, actual)
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    def f1_calc(self, pred, actual):
        self.compute(pred, actual)
        precision = self.precision_calc(pred, actual)
        recall = self.recall_calc(pred, actual)
        if precision + recall == 0.0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def accuracy_calc(self, pred, actual):
        self.compute(pred, actual)
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total
