"""Download Qwen3-1.7B from HF-Mirror into ../model."""

import os

REPO_ID = "Qwen/Qwen3-1.7B"
DOWNLOAD_URL = "https://hf-mirror.com"

os.environ["HF_ENDPOINT"] = DOWNLOAD_URL
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_XET_DATA_PROGRESS_UPDATE_SPEED_SAMPLING_WINDOW"] = "5s"

import logging
import math
import shutil
import sys
import time
from pathlib import Path
from threading import Lock
from unittest.mock import patch

from huggingface_hub import snapshot_download
from huggingface_hub.utils._xet_progress_reporting import XetDownloadProgressReporter

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MAX_WORKERS = 8
WARMUP = 3.0


# ── Formatting ──────────────────────────────────────────────

def format_bytes(value):
    if value is None:
        return "--"

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def format_rate(value):
    return "--" if value is None else f"{format_bytes(value)}/s"


def format_time(seconds):
    if seconds is None:
        return "--"

    hours, remainder = divmod(math.ceil(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"


# ── Layer 1: Statistics ──────────────────────────────────────

class ProgressTracker:
    """Aggregate native Xet reports and estimate stage ETA."""

    def __init__(self):
        self.reports = {}
        self.clocks = {}
        self.closed = set()

    @staticmethod
    def values(report, kind):
        if kind == "N":
            total = report.total_transfer_bytes
            done = report.total_transfer_bytes_completed
            speed = report.total_transfer_bytes_completion_rate
        else:
            total = report.total_bytes
            done = report.total_bytes_completed
            speed = report.total_bytes_completion_rate

        if speed is not None and (
            not math.isfinite(speed) or speed < 0
        ):
            speed = None

        return total, done, speed

    def update(self, group, report):
        if group in self.closed:
            return

        now = time.monotonic()
        previous = self.reports.get(group)

        # Track warm-up and inactivity, not speed itself.
        for kind in ("N", "R"):
            _, done, _ = self.values(report, kind)

            old = (
                self.values(previous, kind)[1]
                if previous is not None else 0
            )

            key = group, kind

            if done < old:
                self.clocks.pop(key, None)

            elif done > old:
                first, last = self.clocks.get(key, (now, now))

                self.clocks[key] = (
                    now if now - last >= WARMUP else first,
                    now,
                )

        self.reports[group] = report

    def stage(self, kind):
        now = time.monotonic()

        rows = [
            (g, *self.values(r, kind))
            for g, r in self.reports.items()
        ]

        active = [row for row in rows if row[0] not in self.closed]
        pending = [(g, t, d, v) for g, t, d, v in active if t > d]
        rates = [v for _, _, _, v in active if v is not None]

        def ready(group, speed):
            first, last = self.clocks.get((group, kind), (now, now))

            return (
                speed is not None and speed > 0
                and now - first >= WARMUP
                and now - last < WARMUP
            )

        # Estimate only the known outstanding workload.
        eta = (
            sum(t - d for _, t, d, _ in pending)
            / sum(v for _, _, _, v in pending)
            if pending and all(ready(g, v) for g, _, _, v in pending)
            else None
        )

        return (
            sum(t for _, t, _, _ in rows),
            sum(d for _, _, d, _ in rows),
            sum(rates) if rates else None,
            eta,
            bool(pending),
        )

    def snapshot(self, status=None):
        if status == "Completed":
            return status, "", 1.0, None, None, 0.0, ""

        nt, nd, vn, tn, transferring = self.stage("N")
        rt, rd, vr, tr, processing = self.stage("R")

        # Prefer actual network progress when available.
        kind, ratio = (
            ("N", nd / nt) if nt else
            ("R", rd / rt) if rt else
            ("", None)
        )

        if transferring:
            phase, eta, source = "Downloading", tn, "N"

        elif processing:
            phase = "Processing" if nt or rd else "Preparing"
            eta, source = tr, "R"

        else:
            phase = "Processing" if self.reports else "Preparing"
            eta, source = None, ""

        if status:
            phase, eta, source = status, None, ""

        return (
            phase, kind,
            min(1.0, ratio) if ratio is not None else None,
            vn, vr, eta, source,
        )


# ── Layer 2: Rendering ───────────────────────────────────────

class ProgressRenderer:
    """Render a single terminal line."""

    WIDTH = 24

    def __init__(self):
        self.tty = sys.stdout.isatty()
        self.last = -math.inf

    def show(self, stats, force=False):
        status, kind, ratio, vn, vr, eta, source = stats
        completed = status == "Completed"

        if not self.tty and not completed:
            return

        now = time.monotonic()

        if not force and not completed and now - self.last < 0.2:
            return

        self.last = now

        percent = (
            f"{math.floor(ratio * 1000) / 10:5.1f}%"
            if ratio is not None else "    --"
        )

        remaining = format_time(eta)

        if eta is not None and eta > 0:
            remaining = f"~{remaining} [{source}]"

        prefix = f"{status:<11} "
        suffix = (
            f" {kind} {percent}"
            f" | Net {format_rate(vn):>12}"
            f" | Local {format_rate(vr):>12}"
            f" | ETA {remaining}"
        )

        columns = shutil.get_terminal_size().columns - 1
        width = max(0, min(self.WIDTH, columns - len(prefix + suffix)))

        filled = int(width * (ratio or 0))
        bar = "█" * filled + "░" * (width - filled)

        line = prefix + bar + suffix

        if not self.tty:
            print(line, flush=True)
            return

        # Compact output when the terminal is too narrow.
        if len(line) > columns:
            line = (
                f"{status} {kind} {percent}"
                f" | ETA {remaining}"
                f" | N {format_rate(vn)}"
                f" | R {format_rate(vr)}"
            )

        sys.stdout.write(
            "\r" + line[:max(1, columns)] + "\033[K"
            + ("\n" if completed else "")
        )
        sys.stdout.flush()

    def clear(self):
        if self.tty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


# ── Layer 3: Downloading ─────────────────────────────────────

class Downloader:
    @staticmethod
    def run():
        options = dict(
            repo_id=REPO_ID,
            endpoint=DOWNLOAD_URL,
            local_dir=MODEL_DIR,
            max_workers=MAX_WORKERS,
            token=False,
        )

        print(f"Model:  {REPO_ID}")
        print(f"Mirror: {DOWNLOAD_URL}")
        print(f"Target: {MODEL_DIR}")
        print("Checking files and cache...", flush=True)

        # Step1: Preview download workload.
        plan = snapshot_download(**options, dry_run=True)

        if not plan:
            raise RuntimeError("Empty download plan.")

        pending = [f for f in plan if f.will_download]
        sizes = [f.file_size for f in pending]
        size = sum(sizes) if all(s is not None for s in sizes) else None

        print(f"Files cached: {len(plan) - len(pending)}/{len(plan)}")
        print(f"Pending: {len(pending)} files")
        print(f"Data size: {format_bytes(size)}", flush=True)

        # Step2: Connect native Xet reports to our renderer.
        tracker = ProgressTracker()
        renderer = ProgressRenderer()
        lock = Lock()

        reporter = XetDownloadProgressReporter
        original_update = reporter.update_progress
        original_close = reporter.close

        def update(group, report, *args, **kwargs):
            original_update(group, report, *args, **kwargs)

            with lock:
                tracker.update(group, report)
                renderer.show(tracker.snapshot())

        def close(group):
            try:
                return original_close(group)
            finally:
                with lock:
                    tracker.closed.add(group)
                    renderer.show(tracker.snapshot())

        try:
            renderer.show(tracker.snapshot())

            # Step3: Download using the original HF implementation.
            with patch.object(reporter, "update_progress", update), \
                 patch.object(reporter, "close", close):

                result = snapshot_download(
                    **options,
                    revision=plan[0].commit_hash,
                )

            # Step4: Verify files before confirming completion.
            renderer.show(tracker.snapshot("Checking"), force=True)

            for file in plan:
                path = MODEL_DIR / file.filename

                if not path.is_file() or (
                    file.file_size is not None
                    and path.stat().st_size != file.file_size
                ):
                    raise RuntimeError(f"Missing or invalid file: {path}")

            renderer.show(tracker.snapshot("Completed"))
            return result

        except BaseException:
            renderer.clear()
            raise


# ── Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.disable(logging.CRITICAL)

    start = time.monotonic()
    result = Downloader.run()

    print(f"Download Time: {format_time(time.monotonic() - start)}")
    print(f"Stored at: {result}")