class ClassificationMetricsCalculator:
    def __init__(self):
        self.true_pos = 0
        self.false_pos = 0
        self.true_neg = 0
        self.false_neg = 0

    def refresh_metrics(self, pred, act):
        if pred == act:
            if pred == 1:
                self.true_pos += 1
            else:
                self.true_neg += 1
        else:
            if pred == 1:
                self.false_pos += 1
            else:
                self.false_neg += 1

    def calculate_precision(self, pred, act):
        self.refresh_metrics(pred, act)
        if self.true_pos + self.false_pos == 0:
            return 0.0
        return self.true_pos / (self.true_pos + self.false_pos)

    def calculate_recall(self, pred, act):
        self.refresh_metrics(pred, act)
        if self.true_pos + self.false_neg == 0:
            return 0.0
        return self.true_pos / (self.true_pos + self.false_neg)

    def calculate_f1(self, pred, act):
        self.refresh_metrics(pred, act)
        precision = self.calculate_precision(pred, act)
        recall = self.calculate_recall(pred, act)
        if precision + recall == 0.0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def calculate_accuracy(self, pred, act):
        self.refresh_metrics(pred, act)
        total = self.true_pos + self.true_neg + self.false_pos + self.false_neg
        if total == 0:
            return 0.0
        return (self.true_pos + self.true_neg) / total
