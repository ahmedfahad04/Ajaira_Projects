class PerformanceEvaluator:
    def __init__(self):
        self.correct_positives = 0
        self.misclassifications = 0
        self.correct_negatives = 0
        self.total_samples = 0

    def update_metrics(self, predictions, actuals):
        for pred, act in zip(predictions, actuals):
            if pred == act:
                if pred == 1:
                    self.correct_positives += 1
                else:
                    self.correct_negatives += 1
            else:
                self.misclassifications += 1
            self.total_samples += 1

    def calculate_precision(self, predictions, actuals):
        self.update_metrics(predictions, actuals)
        if self.correct_positives + self.misclassifications == 0:
            return 0.0
        return self.correct_positives / (self.correct_positives + self.misclassifications)

    def calculate_recall(self, predictions, actuals):
        self.update_metrics(predictions, actuals)
        if self.correct_positives + self.misclassifications == 0:
            return 0.0
        return self.correct_positives / (self.correct_positives + self.misclassifications)

    def calculate_f1(self, predictions, actuals):
        self.update_metrics(predictions, actuals)
        precision = self.calculate_precision(predictions, actuals)
        recall = self.calculate_recall(predictions, actuals)
        if precision + recall == 0.0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def calculate_accuracy(self, predictions, actuals):
        self.update_metrics(predictions, actuals)
        if self.total_samples == 0:
            return 0.0
        return (self.correct_positives + self.correct_negatives) / self.total_samples
