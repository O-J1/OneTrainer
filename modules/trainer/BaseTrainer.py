from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig, prepare_run_name
from modules.util.tensorboard_util import TensorboardProcess, stop_tensorboard, tensorboard_args
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress

import torch


class BaseTrainer(
    TimedActionMixin,
    metaclass=ABCMeta,
):

    tensorboard_subprocess: TensorboardProcess | None

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__()
        self.config = config
        self.tensorboard_subprocess = None
        self.callbacks = callbacks
        self.commands = commands
        self.train_device = torch.device(self.config.train_device)
        self.temp_device = torch.device(self.config.temp_device)
        self.config.run_name = prepare_run_name(
            self.config.training_method,
            self.config.run_name_mode,
            self.config.run_name,
        )

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def end(self):
        pass

    def create_model_loader(self) -> BaseModelLoader:
        return create.create_model_loader(self.config.model_type, self.config.training_method)

    def create_model_setup(self) -> BaseModelSetup:
        return create.create_model_setup(
            self.config.model_type,
            self.train_device,
            self.temp_device,
            self.config.training_method,
            self.config.debug_mode,
        )

    def create_data_loader(self, model: BaseModel, model_setup: BaseModelSetup, train_progress: TrainProgress, is_validation=False):
        return create.create_data_loader(
            self.train_device,
            self.temp_device,
            model,
            self.config.model_type,
            model_setup,
            self.config.training_method,
            self.config,
            train_progress,
            is_validation,
        )

    def create_model_saver(self) -> BaseModelSaver:
        return create.create_model_saver(self.config.model_type, self.config.training_method)

    def create_model_sampler(self, model: BaseModel) -> BaseModelSampler:
        return create.create_model_sampler(
            self.train_device,
            self.temp_device,
            model,
            self.config.model_type,
            self.config.training_method
        )
    def _start_tensorboard(self):
        args = tensorboard_args(self.config)
        self.tensorboard_subprocess = TensorboardProcess(args)

    def _stop_tensorboard(self):
        process, self.tensorboard_subprocess = self.tensorboard_subprocess, None
        stop_tensorboard(process)
