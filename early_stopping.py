import math


# class EarlyStopping:
#     def __init__(self, tolerance=5, min_delta=0):
#
#         self.tolerance = tolerance
#         self.min_delta = min_delta
#         self.counter = 0
#         self.early_stop = False
#
#     def __call__(self, train_loss, validation_loss):
#         if (validation_loss - train_loss) > self.min_delta:
#             self.counter += 1
#             if self.counter >= self.tolerance:
#                 self.early_stop = True

class EarlyStopping:
    def __init__(self, tolerance=5, metric='accuracy'):

        self.tolerance = tolerance
        self.min_best_value = math.inf
        self.max_best_value = 0
        self.last_best = self.min_best_value
        self.metric = metric
        self.counter = 0
        self.early_stop = False

    def __call__(self, monitor):
        assert self.metric == 'loss' or self.metric == 'accuracy'

        if self.metric == 'loss':
            if monitor < self.min_best_value:
                self.last_best = self.min_best_value
                self.min_best_value = monitor
                self.counter = 0

            if monitor > self.min_best_value:
                self.counter += 1

                if self.counter >= self.tolerance:
                    self.early_stop = True
                    return self.early_stop

        if self.metric == 'accuracy':
            if monitor > self.max_best_value:
                self.last_best = self.max_best_value
                self.max_best_value = monitor
                self.counter = 0

            if monitor < self.max_best_value:
                self.counter += 1

                if self.counter >= self.tolerance:
                    self.early_stop = True
                    return self.early_stop

    def get_last_best(self):
        return self.last_best

    def get_best_value(self):
        if self.metric == "loss":
            return self.min_best_value
        if self.metric == "accuracy":
            return self.max_best_value

    def get_counter(self):
        return self.counter


def test():
    e = EarlyStopping(tolerance=3, metric='loss')

    losses = [math.inf, 0.21, 0.10, 0.41, 0.01, 0.3, 0.34, 0.5]
    accs = [0.1, 0.2, 0.30, 0.2, 0.01, 0.7, 0.19, 0.01, 0.12, 0.11]
    t = losses

    for i in range(len(t)):
        res = e(t[i])

        if res:
            print("best val: ", e.get_best_value())
            print("counter: ", e.get_counter())
            print("Early stopping at {} with value= {} ".format(i, t[i]))
            break


if __name__ == "__main__":
    test()
