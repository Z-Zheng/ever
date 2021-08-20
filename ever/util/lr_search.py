from ever.core.logger import TrainLogHook


class PlotLearningRateAndLoss(TrainLogHook):
    def __init__(self, interval_step, save_path):
        super(PlotLearningRateAndLoss, self).__init__(interval_step)
        self.losses = []
        self.lrs = []
        self.save_path = save_path

    def after_iter(self,
                   current_step,
                   loss_dict,
                   learning_rate,
                   num_iters,
                   ):
        self.losses.append(loss_dict['total_loss'])
        self.lrs.append(learning_rate)

    def after_train(self,
                    current_step,
                    loss_dict,
                    learning_rate,
                    num_iters,
                    ):
        import matplotlib.pyplot as plt
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
        plt.savefig(self.save_path)
        plt.close()
