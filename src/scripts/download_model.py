
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
    """Track the download plan; tqdm handles live speed and ETA."""

    def __init__(self, plan):
        self.pending = [f for f in plan if f.will_download]
        self.total_files = len(plan)
        self.cached_files = len(plan) - len(self.pending)
        self.revision = plan[0].commit_hash if plan else None

        sizes = [f.file_size for f in self.pending]
        self.total_bytes = (
            sum(sizes)
            if all(size is not None for size in sizes)
            else None
        )


# ── Layer 2: Rendering ────────────────────────────────────────

class ProgressRenderer:
    """Configure native tqdm formatting."""

    FORMATS = {
        "network": (
            "{desc:<9} {n_fmt}B | {rate_fmt}"
        ),
        "overall": (
            "{desc:<9} {percentage:5.1f}% |{bar:18u}| "
            "{n_fmt}B/{total_fmt}B | {rate_fmt} | ETA {remaining}"
        ),
        "files": (
            "{desc:<9} |{bar:18u}| {n_fmt}/{total_fmt}"
        ),
    }

    UNKNOWN = "{desc:<9} {n_fmt}B processed | {rate_fmt}"

    @classmethod
    def configure(cls, kind, kwargs, tracker):
        known_total = tracker and tracker.total_bytes is not None

        kwargs.update(
            desc=kind.title(),
            bar_format=(
                cls.UNKNOWN
                if kind == "overall" and not known_total
                else cls.FORMATS[kind]
            ),
            ascii=False,
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.3,
            leave=False,
        )

        if kind in ("network", "overall"):
            kwargs["unit_divisor"] = 1024

    @staticmethod
    def show_plan(tracker):
        print(
            f"Files: {tracker.cached_files}/{tracker.total_files} cached, "
            f"{len(tracker.pending)} pending"
        )
        print(f"Data size: {format_bytes(tracker.total_bytes)}")

    @staticmethod
    def show_result(path, elapsed):
        print(f"\nCompleted in {format_time(elapsed)}")
        print(f"Model ready: {path}")


# ── Adapter: Hugging Face -> Tracker -> Renderer ──────────────

class DownloadProgress(tqdm):
    tracker = None

    def __init__(self, *args, **kwargs):
        desc = str(kwargs.get("desc") or "").lower()

        # Identify the three progress streams from Hugging Face.
        if kwargs.get("unit") != "B":
            self.kind = "files"
        elif "reconstruct" in desc:
            self.kind = "overall"
        else:
            self.kind = "network"

        # DownloadProgress -> ProgressRenderer
        ProgressRenderer.configure(
            self.kind, kwargs, self.tracker
        )

        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        data = super().format_dict

        # Apply preflight total for DISPLAY only.
        # Never modify tqdm's actual counters.
        if (
            self.kind == "overall"
            and self.tracker is not None
            and self.tracker.total_bytes is not None
        ):
            expected = self.tracker.total_bytes

            if data["n"] <= expected:
                data["total"] = expected
            else:
                # The preflight estimate no longer matches.
                data["total"] = None
                data["bar_format"] = ProgressRenderer.UNKNOWN

        return data


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
        ProgressRenderer.show_plan(tracker)

        # Early Return if there's no file needing to be downloaded.
        if not tracker.pending:
            return str(MODEL_DIR.resolve())

        # Downloader -> DownloadProgress -> ProgressTracker
        DownloadProgress.tracker = tracker

        # Step4: Start Downloading.
        return snapshot_download(
            **options,
            revision=tracker.revision,
            tqdm_class=DownloadProgress,
        )


# ── Entry point ───────────────────────────────────────────────

def main():
    start = time.monotonic()

    # Avoid warning information disturbing the progress display.
    with logging_redirect_tqdm(
        loggers=[logging.getLogger("huggingface_hub")]
    ):
        result = Downloader.run()  # main entrance

    # All process done, print running time and ending signal.
    elapsed = time.monotonic() - start
    ProgressRenderer.show_result(result, elapsed)


if __name__ == "__main__":
    main()
