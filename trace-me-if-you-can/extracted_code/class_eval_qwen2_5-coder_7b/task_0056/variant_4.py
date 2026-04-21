class ClassificationStats:
    def __init__(self):
        self.positives_correct = 0
        self.negative_correct = 0
        self.positives_incorrect = 0
        self.negative_incorrect = 0

    def update_stats(self, predicted, actual):
        if predicted == actual:
            if predicted == 1:
                self.positives_correct += 1
            else:
                self.negative_correct += 1
        else:
            if predicted == 1:
                self.positives_incorrect += 1
            else:
                self.negative_incorrect += 1

    def get_precision(self, predicted, actual):
        self.update_stats(predicted, actual)
        if self.positives_correct + self.positives_incorrect == 0:
            return 0.0
        return self.positives_correct / (self.positives_correct + self.positives_incorrect)

    def get_recall(self, predicted, actual):
        self.update_stats(predicted, actual)
        if self.positives_correct + self.negative_incorrect == 0:
            return 0.0
        return self.positives_correct / (self.positives_correct + self.negative_incorrect)

    def get_f1(self, predicted, actual):
        self.update_stats(predicted, actual)
        precision = self.get_precision(predicted, actual)
        recall = self.get_recall(predicted, actual)
        if precision + recall == 0.0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def get_accuracy(self, predicted, actual):
        self.update_stats(predicted, actual)
        total = self.positives_correct + self.negative_correct + self.positives_incorrect + self.negative_incorrect
        if total == 0:
            return 0.0
        return (self.positives_correct + self.negative_correct) / total
