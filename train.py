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
    gpu=A10G(count=2),
    mounts=[
        modal.Mount.from_local_python_packages("main"),
        modal.Mount.from_local_dir(local_path="exp", remote_path="/root/exp"),
        modal.Mount.from_local_dir(local_path="ml-data", remote_path="/root/ml-data"),
        modal.Mount.from_local_file(local_path="config.yaml", remote_path="/root/config.yaml"),
        modal.Mount.from_local_file(local_path=".env", remote_path="/root/.env"),
        modal.Mount.from_local_file(local_path="train_script.py", remote_path="/root/train_script.py"),
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
        # add config to command line arguments
        sys.argv.append(f"{config}")

        import subprocess
        subprocess.run(
            ["python", "train_script.py", config],
            stdout=sys.stdout, stderr=sys.stderr,
            check=True,
        )


@stub.local_entrypoint()
def main(config: str = "exp=diffusion_test"):
    model_trainer = ModelTrainer()
    model_trainer.run_cli.remote(config)


if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.run_cli.local(config="exp=diffusion_test")
