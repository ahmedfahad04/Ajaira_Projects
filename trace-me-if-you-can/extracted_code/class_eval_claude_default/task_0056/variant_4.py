class MetricsCalculator:
    def __init__(self):
        self.metrics = [0, 0, 0, 0]  # [tp, fp, fn, tn]

    def update(self, predicted_labels, true_labels):
        for predicted, true in zip(predicted_labels, true_labels):
            index = predicted * 2 + (1 - true)  # Maps (1,1)->0, (1,0)->1, (0,1)->2, (0,0)->3
            if index == 0:  # (1,1) -> tp
                self.metrics[0] += 1
            elif index == 1:  # (1,0) -> fp
                self.metrics[1] += 1
            elif index == 2:  # (0,1) -> fn
                self.metrics[2] += 1
            else:  # (0,0) -> tn
                self.metrics[3] += 1

    @property
    def true_positives(self):
        return self.metrics[0]

    @property
    def false_positives(self):
        return self.metrics[1]

    @property
    def false_negatives(self):
        return self.metrics[2]

    @property
    def true_negatives(self):
        return self.metrics[3]

    def precision(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    def f1_score(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        precision = self.precision(predicted_labels, true_labels)
        recall = self.recall(predicted_labels, true_labels)
        if precision + recall == 0.0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def accuracy(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        total = sum(self.metrics)
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
