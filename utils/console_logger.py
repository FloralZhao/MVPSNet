import logging
import time
import os

__all__ = [
    'ConsoleLogger'
]


class CustomFormatter(logging.Formatter):
    """Custom formatter"""

    DATE = '\033[94m'
    GREEN = '\033[92m'
    WHITE = '\033[0m'
    WARNING = '\033[93m'
    RED = '\033[91m'

    def __init__(self):
        """initializer"""

        orig_fmt = "%(name)s: %(message)s"
        datefmt = "%H:%M:%S"

        super().__init__(orig_fmt, datefmt)

    def format(self, record):
        """format message"""

        color = self.WHITE
        if record.levelno == logging.INFO:
            color = self.GREEN

        if record.levelno == logging.WARN:
            color = self.WARNING

        if record.levelno == logging.ERROR:
            color = self.RED

        self._style._fmt = "{}%(asctime)s {}[%(levelname)s]{} {}: %(message)s".format(
            self.DATE, color, self.DATE, self.WHITE)

        return logging.Formatter.format(self, record)


class ConsoleLogger():
    """Console logger"""

    def __init__(self, phase='train', logfile_dir=None, exp_name=None, abs_logdir=None):

        super().__init__()

        if phase in ['val', 'test', 'eval'] and (logfile_dir is None and abs_logdir is None):
            raise ValueError(f'logfile_dir is None for val / test / eval.')

        self._logger = logging.getLogger('')
        self._logger.setLevel(logging.INFO)

        # ------------------- console logging -------------------
        formatter = CustomFormatter()
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        console_log.setFormatter(formatter)
        self._logger.addHandler(console_log)

        # ------------------- file logging -------------------
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        if abs_logdir is not None:
            self.logfile_dir = abs_logdir
        else:
            if phase == 'train':
                if exp_name is not None:
                    self.logfile_dir = os.path.join('./experiments', exp_name, time_str)
                else:
                    self.logfile_dir = os.path.join('./experiments', time_str)

            else:
                self.logfile_dir = os.path.join(logfile_dir, phase, time_str)

        if not os.path.exists(self.logfile_dir):
            os.makedirs(self.logfile_dir)

        logfile = os.path.join(self.logfile_dir, f'{phase}.log')

        file_log = logging.FileHandler(logfile, mode='a')
        file_log.setLevel(logging.INFO)
        file_log.setFormatter(formatter)
        self._logger.addHandler(file_log)

        # ------------------- save dirs -------------------
        # ------------- checkpoint_dir
        if phase == 'train':
            checkpoint_dir = os.path.join(self.logfile_dir, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.checkpoint_dir = checkpoint_dir

        # ------------- vis_dir
        vis_dir = os.path.join(self.logfile_dir, 'vis')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        self.vis_dir = vis_dir


    def info(self, *args, **kwargs):
        """info"""
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """warning"""
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """error"""
        self._logger.error(*args, **kwargs)
        exit(-1)

    def getLogFolder(self):
        return self.logfile_dir

    def getCheckpointFolder(self):
        return self.checkpoint_dir

    def getVisFolder(self):
        return self.vis_dir
