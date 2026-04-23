#!/usr/bin/env python3
"""SLURM job dashboard with live updates and log viewer.

Usage:
    uv run --extra dashboard python scripts/slurm/dashboard.py
    uv run --extra dashboard python scripts/slurm/dashboard.py --host user@cluster
    uv run --extra dashboard python scripts/slurm/dashboard.py --interval 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import DataTable, Footer, Header, RichLog, Static

SSH_HOST = "fdsreckziegel@pcad.inf.ufrgs.br"
LOGS_DIR = Path("logs")
POLL_INTERVAL = 30


def fetch_jobs(host: str, user: str | None = None) -> list[dict]:
    """Fetch SLURM jobs via SSH + squeue JSON output."""
    user_filter = f"-u {user}" if user else ""
    cmd = f'ssh {host} "squeue {user_filter} --json" 2>/dev/null'
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        return data.get("jobs", [])
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return []


def format_time(seconds: int) -> str:
    """Format seconds into HH:MM:SS."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_time_limit(minutes: int) -> str:
    """Format time limit minutes into readable string."""
    if minutes >= 1440:
        return f"{minutes // 1440}d {(minutes % 1440) // 60:02d}h"
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def fetch_recent_jobs(host: str, limit: int = 20) -> list[dict]:
    """Scan the cluster logs directory for recent job IDs and metadata."""
    cmd = (
        f'ssh {host} "ls -t ~/etno-kgc-preprocessing/logs/*.out 2>/dev/null'
        f' | head -{limit}" 2>/dev/null'
    )
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            filename = line.rsplit("/", 1)[-1]
            # Parse: kg_tupi_arandu-kg_777144.out → job_id=777144
            parts = filename.rsplit("_", 1)
            if len(parts) == 2:
                job_id = parts[1].replace(".out", "")
                name = "_".join(filename.split("_")[:-1])
                jobs.append({"job_id": job_id, "name": name, "file": filename})
        return jobs
    except subprocess.TimeoutExpired:
        return []


class JobTable(DataTable):
    """Table showing SLURM jobs."""

    BINDINGS = [("r", "refresh", "Refresh")]

    def on_mount(self) -> None:
        """Set up table columns."""
        self.add_columns(
            "Job ID", "Name", "Partition", "Node", "State",
            "Time", "Limit", "CPUs",
        )
        self.cursor_type = "row"


class LogViewer(RichLog):
    """Log viewer panel that fetches logs from the cluster via SSH."""

    def __init__(self, ssh_host: str = SSH_HOST, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.ssh_host = ssh_host
        self._remote_home: str | None = None

    def load_log(self, job_id: str) -> None:
        """Fetch and display log files for a job from the cluster.

        Searches for any file matching *{job_id}* in the remote logs/ dir
        and shows the last 100 lines of each.
        """
        self.clear()
        self.write(f"[dim]Fetching logs for job {job_id}...[/dim]")

        try:
            # Find matching log files on the cluster
            find_cmd = (
                f'ssh {self.ssh_host} '
                f'"find ~/etno-kgc-preprocessing/logs -name \'*{job_id}*\' '
                f'-type f | sort" 2>/dev/null'
            )
            result = subprocess.run(
                find_cmd, shell=True, capture_output=True, text=True, timeout=10,
            )
            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

            if not files:
                self.clear()
                self.write(f"[yellow]No log files found for job {job_id}[/yellow]")
                return

            self.clear()
            for remote_path in files:
                filename = remote_path.rsplit("/", 1)[-1]
                self.write(f"[bold cyan]═══ {filename} ═══[/bold cyan]")

                tail_cmd = (
                    f'ssh {self.ssh_host} "tail -100 {remote_path}" 2>/dev/null'
                )
                tail_result = subprocess.run(
                    tail_cmd, shell=True, capture_output=True, text=True, timeout=10,
                )
                if tail_result.returncode == 0:
                    for line in tail_result.stdout.splitlines():
                        self.write(line)
                else:
                    self.write(f"[red]Failed to read {filename}[/red]")
                self.write("")

        except subprocess.TimeoutExpired:
            self.clear()
            self.write(f"[red]SSH timeout fetching logs for job {job_id}[/red]")


class StatusBar(Static):
    """Status bar showing last update time."""

    last_update: reactive[str] = reactive("Never")
    job_count: reactive[int] = reactive(0)

    def render(self) -> str:
        """Render status bar."""
        return (
            f" Jobs: {self.job_count} | "
            f"Last update: {self.last_update} | "
            f"Press [bold]r[/bold] to refresh, [bold]q[/bold] to quit"
        )


class SlurmDashboard(App):
    """SLURM job monitoring dashboard."""

    CSS = """
    #main {
        height: 1fr;
    }
    #job-panel {
        width: 1fr;
        min-width: 60;
    }
    #log-panel {
        width: 2fr;
        border-left: solid $accent;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    JobTable {
        height: 1fr;
    }
    LogViewer {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(
        self, host: str = SSH_HOST, interval: int = POLL_INTERVAL, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        self.ssh_host = host
        self.poll_interval = interval
        self.ssh_user = host.split("@")[0] if "@" in host else None
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Create the layout."""
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="job-panel"):
                yield JobTable()
            with Vertical(id="log-panel"):
                yield LogViewer(ssh_host=self.ssh_host, highlight=True, markup=True)
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Start polling on mount."""
        self.refresh_jobs()
        self._timer = self.set_interval(self.poll_interval, self.refresh_jobs)

    @work(thread=True)
    def refresh_jobs(self) -> None:
        """Fetch active jobs and recent completed jobs."""
        active_jobs = fetch_jobs(self.ssh_host, self.ssh_user)
        recent_jobs = fetch_recent_jobs(self.ssh_host)
        self.call_from_thread(self._update_table, active_jobs, recent_jobs)

    def _update_table(
        self, active_jobs: list[dict], recent_jobs: list[dict]
    ) -> None:
        """Update the job table with active and recent jobs."""
        table = self.query_one(JobTable)
        table.clear()
        active_ids: set[str] = set()

        # Active jobs first
        for job in active_jobs:
            job_id = str(job.get("job_id", ""))
            active_ids.add(job_id)
            name = job.get("name", "")
            partition = job.get("partition", "")
            node = job.get("nodes", "")
            if not isinstance(node, str):
                node = str(node)

            state = job.get("job_state", ["UNKNOWN"])
            if isinstance(state, list):
                state = state[0] if state else "UNKNOWN"

            elapsed = job.get("time", {}).get("elapsed", 0)
            time_str = format_time(elapsed)

            limit = job.get("time", {}).get("limit", {}).get("number", 0)
            limit_str = format_time_limit(limit)

            cpus = str(job.get("cpus", {}).get("number", ""))

            table.add_row(
                job_id, name, partition, node, state,
                time_str, limit_str, cpus,
                key=job_id,
            )

        # Recent completed jobs (from log files)
        for job in recent_jobs:
            job_id = job["job_id"]
            if job_id in active_ids:
                continue
            table.add_row(
                job_id, job["name"], "", "", "COMPLETED",
                "", "", "",
                key=job_id,
            )

        total = len(active_ids) + sum(
            1 for j in recent_jobs if j["job_id"] not in active_ids
        )
        status = self.query_one(StatusBar)
        status.last_update = datetime.now().strftime("%H:%M:%S")
        status.job_count = total

    @on(DataTable.RowHighlighted)
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Load logs when a job row is highlighted."""
        if event.row_key and event.row_key.value:
            log_viewer = self.query_one(LogViewer)
            log_viewer.load_log(str(event.row_key.value))

    def action_refresh(self) -> None:
        """Manual refresh."""
        self.refresh_jobs()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="SLURM job dashboard")
    parser.add_argument(
        "--host", default=SSH_HOST, help=f"SSH host (default: {SSH_HOST})"
    )
    parser.add_argument(
        "--interval", type=int, default=POLL_INTERVAL,
        help=f"Poll interval in seconds (default: {POLL_INTERVAL})",
    )
    args = parser.parse_args()

    app = SlurmDashboard(host=args.host, interval=args.interval)
    app.run()


if __name__ == "__main__":
    main()
