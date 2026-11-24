
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
from threading import Lock

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from huggingface_hub import snapshot_download


MODEL_DIR = Path(__file__).resolve().parent.parent / "modelcopy"
MAX_WORKERS = 8


# ── Formatting Functions ───────────────────────────────────────

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
    """Manage download progress and track the data."""

    def __init__(self, expected_bytes, total_files):
        self.expected_bytes = expected_bytes
        self.total_files = total_files
        self.bars = {}
        self.lock = Lock()
        self.last_refresh = 0.0
        self.finished = False

    def register(self, kind, bar):
        with self.lock:
            self.bars[kind] = bar

    def get_bar(self, kind):
        with self.lock:
            return self.bars.get(kind)

    def should_refresh(self):
        """Limit UI refreshes from concurrent download threads."""
        with self.lock:
            now = time.monotonic()

            if now - self.last_refresh < 0.5:
                return False

            self.last_refresh = now
            return True

    def finish(self):
        """Mark completion only after snapshot_download succeeds."""
        with self.lock:
            self.finished = True

    def snapshot(self):
        """Read the three data: bytes, files and rate."""
        with self.lock:
            finished = self.finished
            bars = self.bars.copy()

        def read(kind):
            bar = bars.get(kind)
            if bar is None:
                return 0, None

            data = bar.format_dict
            return data["n"], data.get("rate")

        received, network_speed = read("network")
        reconstructed, write_speed = read("overall")
        files_done, _ = read("files")

        # Successful download is the authoritative completion signal.
        if finished:
            return {
                "phase": "Completed",
                "completed": (
                    self.expected_bytes
                    if self.expected_bytes is not None
                    else reconstructed
                ),
                "total": self.expected_bytes,
                "network_speed": None,
                "write_speed": None,
                "files_done": self.total_files,
                "files_total": self.total_files,
                "eta": 0,
            }

        # Use reconstructed bytes for overall progress.
        # Network speed is only used for display.
        completed = reconstructed
        total = self.expected_bytes

        if total is not None and (
            total <= 0 or completed > total
        ):
            total = None

        # Standard ETA formula using tqdm's EMA-smoothed rate.
        eta = None
        if total is not None:
            if completed == total:
                eta = 0
            elif write_speed is not None and write_speed > 0:
                eta = (total - completed) / write_speed

        return {
            "phase": "Downloading",
            "completed": completed,
            "total": total,
            "network_speed": network_speed,
            "write_speed": write_speed,
            "files_done": int(files_done),
            "files_total": self.total_files,
            "eta": eta,
        }


# ── Layer 2: Rendering ────────────────────────────────────────

class ProgressRenderer:
    WIDTH = 14

    @classmethod
    def render(cls, data):
        phase = data["phase"]
        completed = data["completed"]
        total = data["total"]

        # Display overall progress based on reconstructed bytes.
        if total is not None and total > 0:
            ratio = min(completed / total, 1.0)
            filled = round(cls.WIDTH * ratio)

            bar = "█" * filled + "░" * (cls.WIDTH - filled)

            progress = (
                f"{bar} {ratio:5.1%}"
                f" | {format_bytes(completed)}/{format_bytes(total)}"
            )
        else:
            progress = f"{format_bytes(completed)} processed"

        files = f"{data['files_done']}/{data['files_total']}"

        # Render the final state after successful completion.
        if phase == "Completed":
            return (
                f"Completed   {progress}"
                f" | Files {files}"
                f" | ETA {format_time(0)}"
            )

        def format_rate(speed):
            return (
                f"{format_bytes(speed)}/s"
                if speed is not None and speed > 0
                else "--"
            )

        return (
            f"Downloading {progress}"
            f" | Net {format_rate(data['network_speed'])}"
            f" | Write {format_rate(data['write_speed'])}"
            f" | Files {files}"
            f" | ETA {format_time(data['eta'])}"
        )


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

        kwargs.update(
            position=0,
            leave=False,
            mininterval=0.5,
            miniters=1,
            smoothing=0.3,
        )

        super().__init__(*args, **kwargs)

        if self.tracker is not None:
            # DownloadProgress -> ProgressTracker
            self.tracker.register(self.kind, self)

            if self.kind == "overall":
                self.refresh()

    def __str__(self):
        if self.kind != "overall" or self.tracker is None:
            return super().__str__()

        # ProgressTracker -> ProgressRenderer
        return ProgressRenderer.render(
            self.tracker.snapshot()
        )

    def display(self, msg=None, pos=None):
        # Only one tqdm object controls the terminal.
        if self.kind == "overall":
            return super().display(msg=msg, pos=pos)

    def clear(self, *args, **kwargs):
        if self.kind == "overall":
            return super().clear(*args, **kwargs)

    def update(self, n=1):
        result = super().update(n)

        # Network and file updates also refresh the main display.
        if (
            self.kind != "overall"
            and self.tracker is not None
            and self.tracker.should_refresh()
        ):
            overall = self.tracker.get_bar("overall")

            if overall is not None:
                overall.refresh()

        return result


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

        """Step2: Preview all files needed to process."""
        plan = snapshot_download(
            **options,
            dry_run=True,
            tqdm_class=SilentProgress,
        )

        pending = [f for f in plan if f.will_download]  # file num
        sizes = [f.file_size for f in pending]  # file sizes

        total = (
            sum(sizes)
            if all(size is not None for size in sizes)
            else None
        )

        cached = len(plan) - len(pending)

        print(f"Files cached: {cached}/{len(plan)}")
        print(f"Pending: {len(pending)} files")
        print(f"Data size: {format_bytes(total)}", flush=True)

        """Early Return if there's no file needing to be downloaded."""
        if not pending:
            return str(MODEL_DIR.resolve())

        """Step3: Initialize the static data tracking."""
        # Downloader -> DownloadProgress -> ProgressTracker
        tracker = ProgressTracker(
            expected_bytes=total,
            total_files=len(plan),
        )
        DownloadProgress.tracker = tracker

        """Step4: Start Downloading."""
        result = snapshot_download(
            **options,
            revision=plan[0].commit_hash,
            tqdm_class=DownloadProgress,
        )

        """Step5: Confirm completion after successful downloading."""
        tracker.finish()

        # Display final progress without depending on further updates.
        tqdm.write(
            ProgressRenderer.render(tracker.snapshot())
        )

        return result


# ── Entry point ───────────────────────────────────────────────

def main():
    start = time.monotonic()

    """Avoid the warning information influences the download process display."""
    with logging_redirect_tqdm(
        loggers=[logging.getLogger("huggingface_hub")]
    ):
        result = Downloader.run()  # main entrance

    """All process done, print running time and ending signal."""
    elapsed = time.monotonic() - start

    print(f"\nCompleted in {format_time(elapsed)}")
    print(f"Model ready: {result}")


if __name__ == "__main__":
    main()
