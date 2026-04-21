class MetricsCalculator:
    def __init__(self):
        self._reset_counters()

    def _reset_counters(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0

    def update(self, predicted_labels, true_labels):
        predictions = list(zip(predicted_labels, true_labels))
        
        self.true_positives += sum(1 for p, t in predictions if p == 1 and t == 1)
        self.false_positives += sum(1 for p, t in predictions if p == 1 and t == 0)
        self.false_negatives += sum(1 for p, t in predictions if p == 0 and t == 1)
        self.true_negatives += sum(1 for p, t in predictions if p == 0 and t == 0)

    def _safe_divide(self, numerator, denominator):
        return 0.0 if denominator == 0 else numerator / denominator

    def precision(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        return self._safe_divide(self.true_positives, self.true_positives + self.false_positives)

    def recall(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        return self._safe_divide(self.true_positives, self.true_positives + self.false_negatives)

    def f1_score(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        precision = self.precision(predicted_labels, true_labels)
        recall = self.recall(predicted_labels, true_labels)
        return self._safe_divide(2 * precision * recall, precision + recall)

    def accuracy(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return self._safe_divide(self.true_positives + self.true_negatives, total)
