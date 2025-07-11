"""
Collection of loggers for controller performance.
"""
from typing import Callable

import wandb
from stable_baselines3.common.logger import KVWriter, Logger, make_output_format

from commonpower.control.logging_utils.callbacks import (
    BaseCallback,
    MARLBaseCallback,
    MARLWandBCallback,
    SafetyCallback,
    WandBSafetyCallback,
)


class BaseLogger:
    def __init__(self, log_dir: str):
        """
        Base class for logging metrics during RL training.

        Args:
            log_dir (str): relative path to logging directory
        """
        self.log_dir = log_dir

    def get_log_dir(self) -> str:
        return self.log_dir

    def log_function(self) -> Callable:
        raise NotImplementedError

    def finish_logging(self) -> None:
        raise NotImplementedError


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir: str, callback: BaseCallback = SafetyCallback):
        """
        Class for using tensorboard logging in single-agent stable-baselines3 algorithms.

        Args:
            log_dir (str): relative path to logging directory
            callback (BaseCallback, optional): object that implements actual logging during training. By defining a \
            customized callback, additional information can be logged (apart from standard metrics like mean_eps_reward)
        """
        super().__init__(log_dir=log_dir)
        self.callback = callback

    def log_function(self) -> BaseCallback:
        """
        Hands over the callback so it can be used by the stable-baselines3 internal logging.

        Returns:
            BaseCallback: callback which is used during training to log additional information

        """
        return self.callback()

    def finish_logging(self) -> None:
        pass


class WandBLogger(BaseLogger):
    def __init__(
        self,
        log_dir: str,
        entity_name: str,
        run_name: str,
        project_name: str = None,
        callback: BaseCallback = WandBSafetyCallback,
        model_save_freq: int = 100,
        verbose: int = 2,
        alg_config: dict = None,
    ):
        """
        Class for using Weights&Biases (wandb) logging in single-agent stable-baselines3 algorithms.

        Args:
            log_dir (str): relative path to logging directory
            entity_name (str): name of the wandb entity to which the runs will be logged
            run_name (str): name under which the run will be displayed in WandB
            project_name (str, optional): name of the wandb project to which the runs will be logged
            callback (BaseCallback, optional): object that implements actual logging during training. By defining a \
            customized callback, additional information can be logged (apart from standard metrics like mean_eps_reward)
            model_save_freq (int, optional): after how many episodes the current model should be logged
            verbose (int, optional): output verbosity
            alg_config (dict, optional): dictionary of algorithm hyperparameters. Can be used to filter runs in wandb
                API
        """

        super().__init__(log_dir=log_dir)
        self.entity_name = entity_name
        self.run_name = run_name
        self.project_name = project_name
        self.callback = callback
        self.alg_config = alg_config
        self.model_save_freq = model_save_freq
        self.verbose = verbose

        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity_name,
            name=self.run_name,
            config=self.alg_config,
            sync_tensorboard=True,
        )
        self.model_save_path = self.log_dir + f"models/{self.run.id}"
        self.log_dir = self.log_dir + f"runs/{self.run.id}"

    def log_function(self) -> BaseCallback:
        """
        Hands over the callback so it can be used by the stable-baselines3 internal logging.

        Returns:
            BaseCallback: callback which is used during training to log additional information

        """
        return self.callback(
            model_save_path=self.model_save_path, model_save_freq=self.model_save_freq, verbose=self.verbose
        )

    def finish_logging(self) -> None:
        """
        Terminates the W&B run.

        Returns:
            None

        """
        wandb.finish()

    @property
    def run_id(self):
        return wandb.run.id


class MARLTensorboardLogger(BaseLogger):
    def __init__(
        self,
        log_dir: str,
        callback: MARLBaseCallback = MARLBaseCallback,
        format_strings: list = ["stdout", "tensorboard"],
    ):
        """
        Class for using tensorboard logging in multi-agent IPPO/MAPPO algorithms from the on-policy repository
        (https://github.com/marlbenchmark/on-policy/blob/main/README.md).

        Args:
            log_dir (str): relative path to logging directory
            callback (MARLBaseCallback): object that implements actual logging during training. By defining a \
            customized callback, additional information can be logged (apart from standard metrics like mean_eps_reward)
            format_strings (list): list of output formats for the SB3 logger
        """
        super().__init__(log_dir=log_dir)
        self.callback = callback
        log_suffix = ""
        output_formats = [make_output_format(f, self.log_dir, log_suffix) for f in format_strings]
        self.log_function = Logger(folder=self.log_dir, output_formats=output_formats)

    def get_callback(self) -> MARLBaseCallback:
        """
        Hands over the callback.

        Returns:
            MARLBaseCallback: callback which is used during training to log additional information

        """
        return self.callback()

    def get_log_function(self) -> Callable:
        """
        Hands over the logger we get from stable-baselines3

        Returns:
            Callable: Logger

        """
        return self.log_function

    def finish_logging(self) -> None:
        pass


class MARLWandBLogger(BaseLogger):
    def __init__(
        self,
        log_dir: str,
        entity_name: str,
        project_name: str = None,
        callback: BaseCallback = MARLWandBCallback,
        format_strings: list = ["stdout", "tensorboard"],
        model_save_freq: int = 100,
        verbose: int = 2,
        alg_config: dict = None,
    ):
        """
        Class for using Weights&Biases (wandb) logging in single-agent stable-baselines3 algorithms
        Args:
            log_dir (str): relative path to logging directory
            entity_name (str): name of the wandb entity to which the runs will be logged
            project_name (str, optional): name of the wandb project to which the runs will be logged
            callback (BaseCallback, optional): object that implements actual logging during training -
                by defining a customized callback, additional information can be logged
                (apart from standard metrics like mean_eps_reward)
            format_strings (list): list of output formats for the SB3 logger
            model_save_freq (int, optional): after how many episodes the current model should be logged
            verbose (int, optional): output verbosity
            alg_config (dict, optional): dictionary of algorithm hyperparameters.
                Can be used to filter runs in wandb API
        """
        super().__init__(log_dir=log_dir)
        self.entity_name = entity_name
        self.project_name = project_name
        self.callback = callback
        self.alg_config = alg_config

        # init WandB
        self.model_save_freq = model_save_freq
        self.verbose = verbose

        self.run = wandb.init(
            project=self.project_name, entity=self.entity_name, config=self.alg_config, sync_tensorboard=True
        )
        self.model_save_path = self.log_dir + f"models/{self.run.id}"
        self.log_dir = self.log_dir + f"runs/{self.run.id}"

        # init logger (importer from SB3)
        log_suffix = ""
        output_formats = [make_output_format(f, self.log_dir, log_suffix) for f in format_strings]
        self.log_function = Logger(folder=self.log_dir, output_formats=output_formats)

    def get_callback(self) -> MARLBaseCallback:
        """
        Hands over the callback.

        Returns:
            MARLBaseCallback: callback which is used during training to log additional information

        """
        return self.callback(
            model_save_path=self.model_save_path, model_save_freq=self.model_save_freq, verbose=self.verbose
        )

    def get_log_function(self) -> Callable:
        """
        Hands over the logger we get from stable-baselines3

        Returns:
            Callable: Logger

        """
        return self.log_function

    def finish_logging(self) -> None:
        """
        Terminates the W&B run.

        Returns:
            None

        """
        wandb.finish()


class WandBOutputFormatSB3(KVWriter):
    """
    Output format for Weights & Biases.

    :param project: W&B project name
    :param name: W&B run name
    :param entity: W&B entity name
    :param config: Configuration dictionary for the run
    """

    def __init__(self, project=None, name=None, entity=None, config=None):
        self.initialized = False
        self.project = project
        self.name = name
        self.entity = entity
        self.config = config

    def _init_wandb(self):
        if not self.initialized:
            # Initialize wandb only if not already initialized
            if wandb.run is None:
                wandb.init(project=self.project, name=self.name, entity=self.entity, config=self.config, reinit=True)
            self.initialized = True

    def write(self, key_values, key_excluded, step=0):
        """
        Write key-values to W&B

        :param key_values: Dictionary of key-values to log
        :param key_excluded: Dictionary of keys to exclude for certain formats
        :param step: Global step value
        """
        self._init_wandb()

        # Filter out excluded keys
        log_data = {}
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded is not None and "wandb" in excluded:
                continue

            # Handle special types
            # if isinstance(value, (Video, Figure, Image, HParam)):
            #     # These require special handling in wandb
            #     if isinstance(value, Video):
            #         log_data[key] = wandb.Video(value.frames.cpu().numpy(), fps=value.fps)
            #     elif isinstance(value, Figure):
            #         log_data[key] = wandb.Image(value.figure)
            #     elif isinstance(value, Image):
            #         log_data[key] = wandb.Image(value.image)
            #     elif isinstance(value, HParam):
            #         # Log hyperparameters to wandb config
            #         for param_key, param_value in value.hparam_dict.items():
            #             wandb.config.update({param_key: param_value}, allow_val_change=True)
            #         # Log metrics
            #         for metric_key, metric_value in value.metric_dict.items():
            #             log_data[f"hparam/{metric_key}"] = metric_value
            # else:
            #     # Regular scalar values
            log_data[key] = value

        # Add global step if available
        if step > 0:
            log_data["global_step"] = step

        # Log to wandb
        if log_data:
            wandb.log(log_data)

    def close(self):
        """
        Close the wandb run
        """
        if self.initialized and wandb.run is not None:
            wandb.finish()


class WandBLoggerSB3(Logger):
    """
    Logger for Weights & Biases integration with Stable Baselines 3.

    :param folder: Log folder
    :param output_formats: List of output formats
    :param project_name: W&B project name
    :param run_name: W&B run name
    :param entity_name: W&B entity name
    :param config: Configuration dictionary for the run
    """

    def __init__(
        self,
        folder=None,
        output_formats=None,
        project_name="MORL-Baselines",
        run_name=None,
        entity_name=None,
        config=None,
    ):
        if output_formats is None:
            output_formats = []

        # Add WandB format
        wandb_format = WandBOutputFormatSB3(project=project_name, name=run_name, entity=entity_name, config=config)
        output_formats.append(wandb_format)

        super().__init__(folder=folder, output_formats=output_formats)

        # Store wandb-specific info
        self.project_name = project_name
        self.run_name = run_name
        self.entity_name = entity_name

    def dump(self, step=0):
        """
        Write all diagnostics from the current iteration to WandB

        :param step: Current global step
        """

        for _format in self.output_formats:
            if isinstance(_format, KVWriter):
                _format.write(self.name_to_value, self.name_to_excluded, step)

        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

    def get_wandb_run(self):
        """
        Get the current wandb run

        :return: The wandb run
        """
        return wandb.run

    def get_log_function(self):
        """
        Get a function that can be used for logging in other contexts

        :return: A logging function
        """

        def log_fn(key, value, step=None):
            self.record(key, value)
            if step is not None:
                self.dump(step)
            else:
                self.dump()

        return log_fn
