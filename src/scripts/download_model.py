
"""Download Qwen3-1.7B from HF-Mirror into ../modelcopy."""

import os

REPO_ID = "Qwen/Qwen3-1.7B"
DOWNLOAD_URL = "https://hf-mirror.com"

os.environ["HF_ENDPOINT"] = DOWNLOAD_URL
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import logging
import math
import time
from pathlib import Path

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from huggingface_hub import snapshot_download


MODEL_DIR = Path(__file__).resolve().parent.parent / "modelcopy"
MAX_WORKERS = 8


# ── Formatting Functions ─────────────────────────────────────

def format_bytes(value):
    if value is None:
        return "--"

    value = float(value)

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def format_time(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "--h:--m:--s"

    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    return f"{hours:02d}h:{minutes:02d}m:{secs:02d}s"


# ── Layer 1: Statistics ───────────────────────────────────────

class ProgressTracker:
    """Use tqdm's native network speed to estimate remaining time."""

    def __init__(self, plan):
        self.pending = [f for f in plan if f.will_download]
        sizes = [f.file_size for f in self.pending]

        self.total_bytes = (
            sum(sizes) if all(s is not None for s in sizes) else None
        )

    def snapshot(self, bar):
        received = bar.n
        speed = bar.format_dict.get("rate")
        total = self.total_bytes

        # Xet transfer bytes may differ from logical file sizes.
        ratio = min(received / total, 0.99) if total else None

        eta = (
            (total - received) / speed
            if total and speed and speed > 0 and received < total
            else None
        )

        return received, speed, ratio, eta


# ── Layer 2: Rendering ────────────────────────────────────────

class ProgressRenderer:
    WIDTH = 14

    @classmethod
    def render(cls, stats):
        received, speed, ratio, eta = stats

        # Approximate progress: actual network total is unknown.
        if ratio is not None:
            filled = round(cls.WIDTH * ratio)
            bar = "█" * filled + "░" * (cls.WIDTH - filled)
            progress = f"~{ratio:.1%}"
        else:
            bar = "░" * cls.WIDTH
            progress = "unknown"

        rate = f"{format_bytes(speed)}/s" if speed else "--"
        remaining = f"~{format_time(eta)}" if eta is not None else "--"

        return (
            f"Downloading {bar} {progress}"
            f" | {format_bytes(received)}"
            f" | {rate}"
            f" | ETA {remaining}"
        )


# ── Adapter: Hugging Face -> Tracker -> Renderer ──────────────

class DownloadProgress(tqdm):
    tracker = None

    def __init__(self, *args, **kwargs):
        desc = str(kwargs.get("desc") or "").lower()

        # Only show the network transfer progress.
        self.is_network = (
            kwargs.get("unit") == "B" and "download" in desc
        )

        kwargs.update(
            position=0,
            leave=False,
            mininterval=0.5,
            smoothing=0.3,
        )

        super().__init__(*args, **kwargs)

    def __str__(self):
        if not self.is_network or self.tracker is None:
            return super().__str__()

        # DownloadProgress -> ProgressTracker -> ProgressRenderer
        return ProgressRenderer.render(
            self.tracker.snapshot(self)
        )

    def display(self, msg=None, pos=None):
        # Other tqdm objects still work, but do not write to terminal.
        if self.is_network:
            return super().display(msg=msg, pos=pos)

    def clear(self, *args, **kwargs):
        if self.is_network:
            return super().clear(*args, **kwargs)


class SilentProgress(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


# ── Layer 3: Downloading ──────────────────────────────────────

class Downloader:
    @staticmethod
    def run():
        """Step1: Prepare the downloading parameters."""
        options = {
            "repo_id": REPO_ID,
            "endpoint": DOWNLOAD_URL,
            "local_dir": MODEL_DIR,
            "max_workers": MAX_WORKERS,
            "token": False,
        }

        print(f"Model:  {REPO_ID}")
        print(f"Mirror: {DOWNLOAD_URL}")
        print(f"Target: {MODEL_DIR}")
        print("Checking files and cache...", flush=True)

        # Step2: Preview all files needed to process.
        plan = snapshot_download(
            **options,
            dry_run=True,
            tqdm_class=SilentProgress,
        )

        # Step3: Initialize the static data tracking.
        tracker = ProgressTracker(plan)
        cached = len(plan) - len(tracker.pending)

        print(f"Files cached: {cached}/{len(plan)}")
        print(f"Pending: {len(tracker.pending)} files")
        print(f"Data size: {format_bytes(tracker.total_bytes)}")

        # Early Return if there's no file needing to be downloaded.
        if not tracker.pending:
            result = str(MODEL_DIR.resolve())
        else:
            # Downloader -> DownloadProgress -> ProgressTracker
            DownloadProgress.tracker = tracker

            # Step4: Start Downloading.
            result = snapshot_download(
                **options,
                revision=plan[0].commit_hash,
                tqdm_class=DownloadProgress,
            )

        # Step5: Check reconstructed model files.
        print("\nChecking reconstructed files...", flush=True)

        for file in plan:
            path = Path(file.local_path)

            if not path.is_file():
                raise RuntimeError(f"Missing file: {path}")

            if file.file_size is not None:
                if path.stat().st_size != file.file_size:
                    raise RuntimeError(f"Invalid file size: {path}")

        return result


# ── Entry Point ───────────────────────────────────────────────

def main():
    start = time.monotonic()

    # Avoid warnings disturbing the download progress display.
    with logging_redirect_tqdm(
        loggers=[logging.getLogger("huggingface_hub")]
    ):
        result = Downloader.run()  # main entrance

    # All process done, print running time and ending signal.
    elapsed = time.monotonic() - start

    print(f"Completed: {format_time(elapsed)}")
    print(f"Model ready: {result}")


if __name__ == "__main__":
    main()
