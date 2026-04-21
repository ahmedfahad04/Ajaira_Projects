class MetricsCalculator:
    CONFUSION_CATEGORIES = {
        (True, True): 'true_positives',
        (True, False): 'false_positives', 
        (False, True): 'false_negatives',
        (False, False): 'true_negatives'
    }

    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0

    def update(self, predicted_labels, true_labels):
        for predicted, true in zip(predicted_labels, true_labels):
            category = self.CONFUSION_CATEGORIES[(bool(predicted), bool(true))]
            current_value = getattr(self, category)
            setattr(self, category, current_value + 1)

    def _calculate_metric(self, numerator_fn, denominator_fn):
        numerator = numerator_fn()
        denominator = denominator_fn()
        return 0.0 if denominator == 0 else numerator / denominator

    def precision(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        return self._calculate_metric(
            lambda: self.true_positives,
            lambda: self.true_positives + self.false_positives
        )

    def recall(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        return self._calculate_metric(
            lambda: self.true_positives,
            lambda: self.true_positives + self.false_negatives
        )

    def f1_score(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        precision = self.precision(predicted_labels, true_labels)
        recall = self.recall(predicted_labels, true_labels)
        if precision + recall == 0.0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def accuracy(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        return self._calculate_metric(
            lambda: self.true_positives + self.true_negatives,
            lambda: self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        )
