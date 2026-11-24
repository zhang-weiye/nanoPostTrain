
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
    """Track network, reconstruction and file progress."""

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
        with self.lock:
            finished = self.finished
            bars = self.bars.copy()

        # Successful download is the authoritative completion signal.
        if finished:
            return {
                "phase": "Completed",
                "completed": self.expected_bytes or 0,
                "total": self.expected_bytes,
                "network_speed": None,
                "write_speed": None,
                "files_done": self.total_files,
                "files_total": self.total_files,
                "eta": 0,
            }

        def read(kind):
            bar = bars.get(kind)
            if bar is None:
                return 0, None

            data = bar.format_dict
            return data["n"], data.get("rate")

        _, network_speed = read("network")
        reconstructed, write_speed = read("overall")
        files_done, _ = read("files")

        total = self.expected_bytes

        if total is not None and (
            total <= 0 or reconstructed > total
        ):
            total = None

        # ETA based on tqdm's smoothed reconstruction rate.
        eta = None

        if total is not None:
            if reconstructed == total:
                eta = 0
            elif write_speed is not None and write_speed > 0:
                eta = (total - reconstructed) / write_speed

        return {
            "phase": "Downloading",
            "completed": reconstructed,
            "total": total,
            "network_speed": network_speed,
            "write_speed": write_speed,
            "files_done": int(files_done),
            "files_total": self.total_files,
            "eta": eta,
        }


# ── Layer 2: Rendering ────────────────────────────────────────

class ProgressRenderer:
    WIDTH = 12

    @classmethod
    def render(cls, data):
        phase = data["phase"]
        completed = data["completed"]
        total = data["total"]

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

        if phase == "Completed":
            return f"Completed   {progress} | Files {files}"

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
            self.tracker.register(self.kind, self)

            if self.kind == "overall":
                self.refresh()

    def __str__(self):
        if self.kind != "overall" or self.tracker is None:
            return super().__str__()

        return ProgressRenderer.render(
            self.tracker.snapshot()
        )

    def display(self, msg=None, pos=None):
        # Only the overall bar writes to the terminal.
        if self.kind == "overall":
            return super().display(msg=msg, pos=pos)

    def clear(self, *args, **kwargs):
        if self.kind == "overall":
            return super().clear(*args, **kwargs)

    def update(self, n=1):
        result = super().update(n)

        # Network and file events can refresh the main bar.
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
        # Step 1: Configure download.
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

        # Step 2: Check remote files and local cache.
        plan = snapshot_download(
            **options,
            dry_run=True,
            tqdm_class=SilentProgress,
        )

        pending = [f for f in plan if f.will_download]
        sizes = [f.file_size for f in pending]

        total = (
            sum(sizes)
            if all(size is not None for size in sizes)
            else None
        )

        cached = len(plan) - len(pending)

        print(f"Files cached: {cached}/{len(plan)}")
        print(f"Pending: {len(pending)} files")
        print(f"Data size: {format_bytes(total)}", flush=True)

        if not pending:
            print("All files already cached.")
            return str(MODEL_DIR.resolve())

        # Step 3: Initialize progress statistics.
        tracker = ProgressTracker(
            expected_bytes=total,
            total_files=len(plan),
        )

        DownloadProgress.tracker = tracker

        # Step 4: Download and reconstruct files.
        result = snapshot_download(
            **options,
            revision=plan[0].commit_hash,
            tqdm_class=DownloadProgress,
        )

        # Step 5: Confirm completion.
        tracker.finish()
        tqdm.write(ProgressRenderer.render(tracker.snapshot()))

        return result


# ── Entry Point ───────────────────────────────────────────────

def main():
    start = time.monotonic()

    with logging_redirect_tqdm(
        loggers=[logging.getLogger("huggingface_hub")]
    ):
        result = Downloader.run()

    elapsed = time.monotonic() - start

    print(f"\nCompleted in {format_time(elapsed)}")
    print(f"Model ready: {result}")


if __name__ == "__main__":
    main()
