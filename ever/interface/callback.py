class Callback:
    def __init__(
            self,
            epoch_interval: int,
            only_master: bool,
            prior: int = 100,
            before_train=False,
            after_train=False,
    ):
        self._epoch_interval = epoch_interval
        self._only_master = only_master
        self._launcher = None
        self._prior = prior

        self.before_train = before_train
        self.after_train = after_train

    def name(self):
        return ''

    def func(self):
        return NotImplemented

    @property
    def interval(self) -> int:
        return self._epoch_interval

    @property
    def only_master(self) -> bool:
        return self._only_master

    @property
    def prior(self) -> int:
        return self._prior

    @property
    def launcher(self):
        return self._launcher

    def set_launcher(self, launcher):
        self._launcher = launcher

    @property
    def model(self):
        return self._launcher.model

    @property
    def model_without_ddp(self):
        return self._launcher.model_without_ddp

    @property
    def model_dir(self):
        return self._launcher.model_dir

    @property
    def global_step(self):
        return self._launcher.global_step

    @property
    def learning_rate(self):
        return self._launcher.lr

    @property
    def logger(self):
        return self._launcher.logger

    def info(self, msg):
        self._launcher.info(msg)

    def save_model(self, filename=None):
        self._launcher.save_model(filename)


class SaveCheckpointCallback(Callback):
    def __init__(self, epoch_interval: int):
        super().__init__(
            epoch_interval=epoch_interval, only_master=True, prior=0,
            before_train=False,
            after_train=True
        )

    def func(self):
        self.launcher.checkpoint.save()

    def name(self):
        return 'SaveCheckpoint'


class EvaluationCallback(Callback):
    def __init__(
            self,
            dataloader,
            epoch_interval: int,
            only_master: bool,
            after_train=True,
            config=None
    ):
        super().__init__(epoch_interval=epoch_interval, only_master=only_master,
                         before_train=False,
                         after_train=after_train)

        self._dataloader = dataloader
        self._config = config

    def func(self):
        self.launcher.evaluate(self._dataloader, config=self._config)

    def name(self):
        return 'Evaluation'
