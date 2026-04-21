class ModelPerformance:
    def __init__(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    def update_values(self, predicted, actual):
        if predicted == actual:
            if predicted == 1:
                self.true_positive += 1
            else:
                self.true_negative += 1
        else:
            if predicted == 1:
                self.false_positive += 1
            else:
                self.false_negative += 1

    def compute_precision(self, predicted, actual):
        self.update_values(predicted, actual)
        if self.true_positive + self.false_positive == 0:
            return 0.0
        return self.true_positive / (self.true_positive + self.false_positive)

    def compute_recall(self, predicted, actual):
        self.update_values(predicted, actual)
        if self.true_positive + self.false_negative == 0:
            return 0.0
        return self.true_positive / (self.true_positive + self.false_negative)

    def compute_f1(self, predicted, actual):
        self.update_values(predicted, actual)
        precision = self.compute_precision(predicted, actual)
        recall = self.compute_recall(predicted, actual)
        if precision + recall == 0.0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def compute_accuracy(self, predicted, actual):
        self.update_values(predicted, actual)
        total = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        if total == 0:
            return 0.0
        return (self.true_positive + self.true_negative) / total
