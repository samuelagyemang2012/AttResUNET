import math


class ModelCheckpoint:
    def __init__(self, metric='loss'):

        self.min_best_value = math.inf
        self.max_best_value = 0
        self.last_best = 0
        self.metric = metric
        self.found_best = False

    def __call__(self, monitor):

        assert self.metric == 'loss' or self.metric == 'accuracy'

        if self.metric == "loss":
            if monitor < self.min_best_value:
                self.last_best = self.min_best_value
                self.min_best_value = monitor

                self.found_best = True
            else:
                self.found_best = False

            return self.found_best

        if self.metric == "accuracy":
            if monitor > self.max_best_value:
                self.last_best = self.max_best_value
                self.max_best_value = monitor

                self.found_best = True
            else:
                self.found_best = False

            return self.found_best

    def get_last_best(self):
        return self.last_best

    def get_best_value(self):
        if self.metric == "loss":
            return self.min_best_value
        if self.metric == "accuracy":
            return self.max_best_value


def test():
    m = ModelCheckpoint(metric='loss')

    losses = [math.inf, 0.21, 0.10, 0.41, 0.01]
    accs = [0.1, 0.2, 0.30, 0.2, 0.01]

    for l in losses:
        res = m(l)
        if res:
            print("new model found")
            print("val improved from {:.6f} to {:.4f}".format(m.get_last_best(), l))
            print()
        else:
            print("no new model")
            print("last best is " + str(m.get_best_value()))
            print()


# if __name__ == "__main__":
    # test()
