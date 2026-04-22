# /// script
# dependencies = [
#   "huggingface-hub",
#   "inquirer",
#   "requests",
#   "tqdm",
#   "typer"
# ]
# ///
# Download this gist using wget and https://gist.githubusercontent.com/gaardhus/f49f8fd218e0b5ceb2016b93518278ae/raw/hf_gguf_downloader.py

from pathlib import Path
from typing import Optional

import inquirer
import requests
from huggingface_hub import HfApi, hf_hub_url
from tqdm import tqdm
from typer import Typer

app = Typer()


@app.command()
def get_gguf_model(
    repo_id: str, token: Optional[str] = None, output_dir: Optional[Path] = None
):
    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id)

    gguf_models = [file for file in repo_files if file.endswith(".gguf")]

    gguf_files: list[tuple[str, str]] = []

    for model in gguf_models:
        metadata = api.get_hf_file_metadata(url=hf_hub_url(repo_id, model))
        file_size = (
            f"{metadata.size / 1000**3:.2f}GB" if metadata.size else "Unknown Size"
        )
        gguf_files.append((model, file_size))

    longest_file_name = len(max(gguf_files, key=lambda x: len(x[0]))[0])

    answers = inquirer.prompt(
        [
            inquirer.List(
                name="model",
                message="Which model do you want to download?",
                choices=[
                    (f"{file:{longest_file_name + 4}} {size}", file)
                    for file, size in gguf_files
                ],
            )
        ]
    )

    model_file = answers["model"]

    response = requests.get(
        f"https://huggingface.co/{repo_id}/resolve/main/{model_file}?download=true",
        stream=True,
    )

    if output_dir:
        model_file = output_dir / model_file

    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            with open(model_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


if __name__ == "__main__":
    app()
