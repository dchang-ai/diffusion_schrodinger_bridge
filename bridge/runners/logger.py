
from pytorch_lightning.loggers import CSVLogger as _CSVLogger


class Logger:
    def log_metrics(self, metric_dict, step=None, save=False):
        pass

    def log_hparams(self, hparams_dict):
        pass


class CSVLogger(Logger):

    def __init__(self, directory='./', name='logs', save_stride=1):
        self.logger = _CSVLogger(directory, name=name)
        self.count = 0
        self.stride = save_stride

    def log_metrics(self, metrics, step=None, save=False):
        self.count += 1
        self.logger.log_metrics(metrics, step=step)
        # `save()` flushes the buffered rows to metrics.csv and clears them.
        if self.count % self.stride == 0:
            self.logger.save()

        if self.count > self.stride * 10:
            self.count = 0

        if save:
            self.logger.save()

    def log_hparams(self, hparams_dict):
        self.logger.log_hyperparams(hparams_dict)
        self.logger.save()


class NeptuneLogger(Logger):
    def __init__(self, project_name, api_key, save_folder='./'):
        # Imported lazily: the neptune client is an optional extra.
        from pytorch_lightning.loggers import NeptuneLogger as _NeptuneLogger

        self.directory = save_folder
        self.logger = _NeptuneLogger(api_key=api_key, project=project_name)

    def log_metrics(self, metrics, step=None, save=False):
        self.logger.log_metrics(metrics, step=step)

    def log_hparams(self, hparams_dict):
        self.logger.log_hyperparams(hparams_dict)
