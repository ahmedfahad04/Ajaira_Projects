class MetricsCalculator:
    def __init__(self):
        self.confusion_matrix = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    def update(self, predicted_labels, true_labels):
        mapping = {(1, 1): 'tp', (1, 0): 'fp', (0, 1): 'fn', (0, 0): 'tn'}
        for predicted, true in zip(predicted_labels, true_labels):
            key = mapping[(predicted, true)]
            self.confusion_matrix[key] += 1

    def precision(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        denominator = self.confusion_matrix['tp'] + self.confusion_matrix['fp']
        return 0.0 if denominator == 0 else self.confusion_matrix['tp'] / denominator

    def recall(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        denominator = self.confusion_matrix['tp'] + self.confusion_matrix['fn']
        return 0.0 if denominator == 0 else self.confusion_matrix['tp'] / denominator

    def f1_score(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        precision = self.precision(predicted_labels, true_labels)
        recall = self.recall(predicted_labels, true_labels)
        if precision + recall == 0.0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def accuracy(self, predicted_labels, true_labels):
        self.update(predicted_labels, true_labels)
        total = sum(self.confusion_matrix.values())
        if total == 0:
            return 0.0
        return (self.confusion_matrix['tp'] + self.confusion_matrix['tn']) / total
