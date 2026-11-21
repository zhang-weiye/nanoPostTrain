"""Download Qwen3-1.7B into ../model relative to this script."""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import enable_progress_bars, tqdm


class DownloadProgress(tqdm):
    def __init__(self, *args, **kwargs):
        bar_format = kwargs.get("bar_format")
        if kwargs.get("unit") == "B" and bar_format and "{total_fmt" in bar_format:
            kwargs["bar_format"] = bar_format + " [ETA {remaining}]"
        super().__init__(*args, **kwargs)


def main() -> None:
    repo_id = "Qwen/Qwen3-1.7B"
    model_dir = Path(__file__).resolve().parent.parent / "model"
    enable_progress_bars()
    print(f"Checking {repo_id} on Hugging Face…", file=sys.stderr, flush=True)
    snapshot_download(repo_id=repo_id, local_dir=model_dir, tqdm_class=DownloadProgress)
    print(f"Model files ready in {model_dir}")


if __name__ == "__main__":
    main()
