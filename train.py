import modal
from modal import Stub, Image, method
from modal.gpu import A10G, A100, T4, L4
import sys

stub = Stub("wb-ml-tiny-audio-diffusion")

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg"]).run_commands(
        "pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "pytorch-lightning==1.7.7",
        "torchmetrics==0.11.4",
        "python-dotenv",
        "hydra-core",
        "hydra-colorlog",
        "wandb",
        "auraloss",
        "yt-dlp",
        "datasets",
        "pyloudnorm",
        "einops",
        "omegaconf",
        "rich",
        "plotly",
        "librosa",
        "transformers",
        "eng-to-ipa",
        "ema-pytorch",
        "py7zr",
        "notebook",
        "matplotlib",
        "ipykernel",
        "gradio",
        "audio-diffusion-pytorch==0.1.3",
        "audio-encoders-pytorch",
        "audio-data-pytorch",
        "quantizer-pytorch",
        "difformer-pytorch",
        "a-transformers-pytorch",
        "deepspeed",
    )
)

@stub.cls(
    image=image,
    gpu=A10G(count=1),
    mounts=[
        modal.Mount.from_local_python_packages("main"),
        modal.Mount.from_local_dir(local_path="exp", remote_path="/root/exp"),
        modal.Mount.from_local_dir(local_path="ml-data", remote_path="/root/ml-data"),
        modal.Mount.from_local_file(local_path="config.yaml", remote_path="/root/config.yaml"),
        modal.Mount.from_local_file(local_path=".env", remote_path="/root/.env"),
    ],
    timeout=60 * 60, # 1 hour
    cpu=1,
)
class ModelTrainer:
    def __enter__(self):
        # Clear system argvars (prevent errors with cli)
        sys.argv = sys.argv[:1]

    @method()
    def run_cli(self, config: str = "exp=diffusion_test"):

        import os
        import dotenv
        import hydra
        import pytorch_lightning as pl
        from main import utils
        from omegaconf import DictConfig, open_dict
        # import torch # use if direct checkpoint load required (see line 87)

        # Load environment variables from `.env`.
        dotenv.load_dotenv(override=True)
        log = utils.get_logger(__name__)

        @hydra.main(config_path="/root/", config_name="config.yaml", version_base=None)
        def main(config: DictConfig) -> None:

            # Logs config tree
            utils.extras(config)

            # Apply seed for reproducibility
            pl.seed_everything(config.seed)

            # Initialize datamodule
            log.info(f"Instantiating datamodule <{config.datamodule._target_}>.")
            datamodule = hydra.utils.instantiate(config.datamodule, _convert_="partial")

            # Initialize model
            log.info(f"Instantiating model <{config.model._target_}>.")
            model = hydra.utils.instantiate(config.model, _convert_="partial")

            # Initialize all callbacks (e.g. checkpoints, early stopping)
            callbacks = []

            # If save is provided add callback that saves and stops, to be used with +ckpt
            if "save" in config:
                # Ignore loggers and other callbacks
                with open_dict(config):
                    config.pop("loggers")
                    config.pop("callbacks")
                    config.trainer.num_sanity_val_steps = 0
                attribute, path = config.get("save"), config.get("ckpt_dir")
                filename = os.path.join(path, f"{attribute}.pt")
                callbacks += [utils.SavePytorchModelAndStopCallback(filename, attribute)]

            if "callbacks" in config:
                for _, cb_conf in config["callbacks"].items():
                    if "_target_" in cb_conf:
                        log.info(f"Instantiating callback <{cb_conf._target_}>.")
                        callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="partial"))

            # Initialize loggers (e.g. wandb)
            loggers = []
            if "loggers" in config:
                for _, lg_conf in config["loggers"].items():
                    if "_target_" in lg_conf:
                        log.info(f"Instantiating logger <{lg_conf._target_}>.")
                        # Sometimes wandb throws error if slow connection...
                        logger = utils.retry_if_error(
                            lambda: hydra.utils.instantiate(lg_conf, _convert_="partial")
                        )
                        loggers.append(logger)

            # Initialize trainer
            log.info(f"Instantiating trainer <{config.trainer._target_}>.")
            trainer = hydra.utils.instantiate(
                config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
            )

            # Send some parameters from config to all lightning loggers
            log.info("Logging hyperparameters!")
            utils.log_hyperparameters(
                config=config,
                model=model,
                datamodule=datamodule,
                trainer=trainer,
                callbacks=callbacks,
                logger=loggers,
            )

            # Train with checkpoint if present, otherwise from start
            if "ckpt" in config:
                ckpt = config.get("ckpt")
                log.info(f"Starting training from {ckpt}")
                trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)

                # # Alternative model load method
                # # Use if loading from checkpoint with pl trainer causes GPU memory spike (CUDA out of memory).
                # checkpoint = torch.load(config.get("ckpt"), map_location='cpu')['state_dict']
                # model.load_state_dict(checkpoint)
                # trainer.fit(model=model, datamodule=datamodule)
            else:
                log.info("Starting training.")
                trainer.fit(model=model, datamodule=datamodule)

            # Make sure everything closed properly
            log.info("Finalizing!")
            utils.finish(
                config=config,
                model=model,
                datamodule=datamodule,
                trainer=trainer,
                callbacks=callbacks,
                logger=loggers,
            )

            # Print path to best checkpoint
            if (
                not config.trainer.get("fast_dev_run")
                and config.get("train")
                and not config.get("save")
            ):
                log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")
        
        # add config to command line arguments
        sys.argv.append(f"{config}")
        
        main()



@stub.local_entrypoint()
def main(config: str = "exp=diffusion_test"):
    model_trainer = ModelTrainer()
    model_trainer.run_cli.remote(config)


if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.run_cli.local(config="exp=diffusion_test")
