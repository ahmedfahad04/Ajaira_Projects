from collections import Counter

class MetricsCalculator:
    def __init__(self):
        self.counts = Counter()

    def update(self, predicted_labels, true_labels):
        for predicted, true in zip(predicted_labels, true_labels):
            self.counts[(predicted, true)] += 1

    def _get_count(self, predicted, true):
        return self.counts.get((predicted, true), 0)

    def precision(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        tp = self._get_count(1, 1)
        fp = self._get_count(1, 0)
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def recall(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        tp = self._get_count(1, 1)
        fn = self._get_count(0, 1)
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def f1_score(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        precision = self.precision(predicted_labels, true_labels)
        recall = self.recall(predicted_labels, true_labels)
        if precision + recall == 0.0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def accuracy(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        tp = self._get_count(1, 1)
        tn = self._get_count(0, 0)
        total = sum(self.counts.values())
        if total == 0:
            return 0.0
        return (tp + tn) / total
