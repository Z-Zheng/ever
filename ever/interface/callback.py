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
        self._context = None
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
    def train_context(self):
        return self._context

    def set_train_context(self, context):
        self._context = context

    @property
    def model(self):
        return self._context.model

    @property
    def model_without_ddp(self):
        return self._context.model_without_ddp

    @property
    def unwrapped_model(self):
        return self._context.unwrapped_model

    @property
    def model_dir(self):
        return self._context.model_dir

    @property
    def global_step(self):
        return self._context.global_step

    @property
    def learning_rate(self):
        return self._context.lr

    @property
    def logger(self):
        return self._context.logger

    def info(self, msg):
        self._context.info(msg)

    def save_model(self, filename=None):
        self._context.save_model(filename)


class SaveCheckpointCallback(Callback):
    def __init__(self, epoch_interval: int, max_keep=None):
        super().__init__(
            epoch_interval=epoch_interval, only_master=True, prior=0,
            before_train=False,
            after_train=True
        )
        self.max_keep = max_keep

    def func(self):
        self.train_context.checkpoint.save()
        self.train_context.checkpoint.cleanup_checkpoints(self.max_keep)

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
        self.train_context.evaluate(self._dataloader, config=self._config)

    def name(self):
        return 'Evaluation'
