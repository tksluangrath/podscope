"""Terminal UI landing screen for the `podscope` command.

Launches instantly (no Spark/Ollama/spaCy import at module level) and shells
out to `src/run.py` in a subprocess for the actual pipeline work -- see
run_pipeline() for why subprocess instead of a direct import.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Input, ProgressBar, Static

Path("data/logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename="data/logs/tui.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

YOUTUBE_PATTERNS = [
    r"^https://(www\.)?youtube\.com/watch\?.*v=[\w-]+",
    r"^https://youtu\.be/[\w-]+",
    r"^https://(www\.)?youtube\.com/playlist\?.*list=[\w-]+",
]


def validate_url(url: str) -> tuple[bool, str]:
    """Returns (is_valid, kind_or_error). kind is 'video' or 'playlist'."""
    url = url.strip()
    if not url:
        return False, "Please enter a YouTube URL"
    if any(re.match(p, url) for p in YOUTUBE_PATTERNS):
        return True, "playlist" if "list=" in url else "video"
    return False, "Must be a youtube.com or youtu.be URL"


class PipelineProgress(Message):
    def __init__(self, line: str) -> None:
        self.line = line
        super().__init__()


class PipelineComplete(Message):
    def __init__(self, url: str, video_id: str | None) -> None:
        self.url = url
        self.video_id = video_id
        super().__init__()


class PipelineFailed(Message):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__()


class HistoryScreen(Screen):
    """State 7 -- lists processed videos. Reads Iceberg lazily, on_mount only."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("q", "app.pop_screen", "back"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("Loading history...", id="history-status", classes="muted")
        yield DataTable(id="history-table")
        yield Footer()

    def on_mount(self) -> None:
        self.load_history()

    @work(thread=True)
    def load_history(self) -> None:
        # Imported here, not at module level -- avoids JVM init on TUI launch.
        from src import db

        table = self.query_one("#history-table", DataTable)
        status = self.query_one("#history-status", Static)
        try:
            spark = db.build_spark("data/iceberg")
            try:
                rows = (
                    spark.table("local.db.videos")
                    .orderBy("processed_at", ascending=False)
                    .collect()
                )
            finally:
                spark.stop()
        except Exception:
            log.exception("Failed to load history")
            self.app.call_from_thread(status.update, "Could not load history -- see data/logs/tui.log")
            return

        def render() -> None:
            table.add_columns("Title", "Processed At", "Video ID")
            for r in rows:
                table.add_row(r["title"] or "(untitled)", str(r["processed_at"]), r["video_id"])
            status.update(f"{len(rows)} video(s) processed" if rows else "No videos processed yet")

        self.app.call_from_thread(render)


class PodscoreApp(App):
    """Root app -- owns all state transitions for the landing screen."""

    TITLE = "podscope"
    BINDINGS = [
        Binding("h", "show_history", "history"),
        Binding("q", "quit", "quit", show=True),
        Binding("ctrl+c", "quit", "quit", show=False),
        Binding("escape", "cancel", "cancel", show=False),
    ]

    # Catppuccin Mocha -- matches github.com/santifer/career-ops' dashboard theme.
    DEFAULT_CSS = """
    Screen {
        align: center middle;
        background: #1e1e2e;
    }
    #shell {
        width: auto;
        max-width: 80;
        height: auto;
        padding: 1 2;
    }
    #header-bar {
        width: 100%;
        height: auto;
        background: #313244;
        color: #cdd6f4;
        padding: 0 2;
    }
    #header-bar .title { color: #cba6f7; text-style: bold; width: 1fr; }
    #header-bar .subtitle { color: #a6adc8; width: auto; text-align: right; }
    #input-row, #progress-panel, #result-panel, #error-panel {
        background: #313244;
        padding: 1 2;
        margin-top: 1;
    }
    .accent { color: #cba6f7; }
    .muted { color: #a6adc8; }
    .error { color: #f38ba8; }
    .success { color: #a6e3a1; }
    #buttons Button { margin-right: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._url_kind: str | None = None
        self._current_url: str | None = None
        self._start_time: float = 0.0
        self._phase: str = ""
        self._timer = None

    def compose(self) -> ComposeResult:
        with Vertical(id="shell"):
            with Horizontal(id="header-bar"):
                yield Static("PODSCOPE", classes="title")
                yield Static("multi-video nlp pipeline · v0.1.0", classes="subtitle")
            with Vertical(id="idle-panel"):
                yield Static("Paste a YouTube URL below, then press Enter", classes="muted")
                with Vertical(id="input-row"):
                    yield Input(placeholder="https://www.youtube.com/watch?v=...", id="url-input")
                    yield Static("", id="input-error", classes="error")
                with Horizontal(id="buttons"):
                    yield Button("Analyze", id="btn-analyze", variant="primary")
                    yield Button("History", id="btn-history")
                    yield Button("Quit", id="btn-quit")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#url-input", Input).focus()

    def action_show_history(self) -> None:
        self.push_screen(HistoryScreen())

    async def action_cancel(self) -> None:
        if self.query("#progress-panel"):
            self.workers.cancel_all()
            await self._reset_to_idle()

    # -- IDLE -> VALIDATING ---------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        await self._try_submit(event.value)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-analyze":
            await self._try_submit(self.query_one("#url-input", Input).value)
        elif event.button.id == "btn-history":
            self.action_show_history()
        elif event.button.id == "btn-quit":
            self.exit()
        elif event.button.id == "btn-cancel":
            self.workers.cancel_all()
            await self._reset_to_idle()
        elif event.button.id == "btn-retry":
            await self._reset_to_idle()
        elif event.button.id == "btn-analyze-another":
            await self._reset_to_idle()

    async def _try_submit(self, url: str) -> None:
        error = self.query_one("#input-error", Static)
        is_valid, kind = validate_url(url)
        if not is_valid:
            error.update(kind)
            return
        error.update("")
        if kind == "playlist":
            # ponytail: run.py has no playlist ingestion path today (only
            # --url for one video or --urls-file for a pre-built list) --
            # building a fake per-episode progress UI (spec State 4) with
            # nothing behind it would just be decoration. Point the user at
            # the real single-video path instead of pretending this works.
            error.update("Playlists aren't wired up yet -- paste one video URL at a time.")
            return
        self._url_kind = kind
        self._current_url = url
        await self._start_running(url)

    # -- RUNNING ----------------------------------------------------------

    async def _start_running(self, url: str) -> None:
        shell = self.query_one("#shell", Vertical)
        await self.query_one("#idle-panel", Vertical).remove()

        self._start_time = time.monotonic()
        self._phase = "Starting up"
        self._last_download_ts = 0.0
        panel = Vertical(
            Static(f'Processing: "{url}"'),
            Static("", id="step-line", classes="accent"),
            ProgressBar(id="progress-bar", show_eta=False),
            Horizontal(Button("Cancel", id="btn-cancel")),
            id="progress-panel",
        )
        await shell.mount(panel)
        self.query_one("#progress-bar", ProgressBar).update(total=None)
        self._render_step_line()
        self._timer = self.set_interval(1.0, self._render_step_line)
        self.run_pipeline(url)

    def _render_step_line(self) -> None:
        try:
            line = self.query_one("#step-line", Static)
        except Exception:
            return
        now = time.monotonic()
        if self._phase == "Downloading audio" and now - self._last_download_ts > 3:
            self._phase = "Transcribing & analyzing (this can take several minutes)"
        elapsed = int(now - self._start_time)
        line.update(f"⠋ {self._phase}  ({elapsed // 60}:{elapsed % 60:02d} elapsed)")

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    @work(exclusive=True, thread=True)
    def run_pipeline(self, url: str) -> None:
        # Imported here, not at module level -- subprocess isolates the
        # Spark JVM / Ollama calls in a child process so the TUI's own
        # event loop never blocks on their ~8s startup cost.
        import subprocess
        import sys

        def peek_video_id() -> str | None:
            try:
                from src import ingest

                return ingest.peek_video_id(url)
            except Exception:
                log.exception("peek_video_id failed for %s", url)
                return None

        tail: list[str] = []
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "src.run", "--url", url],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                line = line.rstrip()
                tail.append(line)
                tail[:] = tail[-20:]
                self.post_message(PipelineProgress(line))
            process.wait()
            if process.returncode == 0:
                self.post_message(PipelineComplete(url, peek_video_id()))
            else:
                log.error(
                    "run.py exited with code %s. Last output:\n%s",
                    process.returncode,
                    "\n".join(tail),
                )
                self.post_message(PipelineFailed(f"Pipeline exited with code {process.returncode}"))
        except Exception as e:
            log.exception("Pipeline error")
            self.post_message(PipelineFailed(str(e)))

    def on_pipeline_progress(self, message: PipelineProgress) -> None:
        # run.py's own output is raw print()/Spark/yt-dlp log noise, not a
        # step protocol -- classify the few lines that mean something
        # instead of surfacing truncated garbage to the user. Full raw
        # output is still captured in data/logs/tui.log for diagnosis.
        text = message.line
        if "[download]" in text:
            self._phase = "Downloading audio"
            self._last_download_ts = time.monotonic()
        elif "already processed" in text:
            self._phase = "Already processed -- finishing up"
        self._render_step_line()

    # -- COMPLETE / ERROR ---------------------------------------------------

    async def on_pipeline_complete(self, message: PipelineComplete) -> None:
        self._stop_timer()
        await self.query_one("#progress-panel").remove()
        shell = self.query_one("#shell", Vertical)
        await shell.mount(
            Vertical(
                Static(f'✓ Complete — "{message.url}"', classes="success"),
                Static("", id="result-body", classes="muted"),
                Horizontal(
                    Button("Analyze Another", id="btn-analyze-another"),
                    Button("View in History", id="btn-history"),
                    Button("Quit", id="btn-quit"),
                ),
                id="result-panel",
            )
        )
        if message.video_id:
            self.load_result_stats(message.video_id)
        else:
            self.query_one("#result-body", Static).update("Video processed.")

    @work(thread=True)
    def load_result_stats(self, video_id: str) -> None:
        from src import db

        try:
            spark = db.build_spark("data/iceberg")
            try:
                segs = db.read_segments(spark, video_id)
                agg = segs.selectExpr(
                    "count(*) as n",
                    "count(distinct topic_label) as topics",
                    "avg(compression_ratio) as avg_cr",
                    "avg(semantic_similarity) as avg_sim",
                ).collect()[0]
                divergent = (
                    segs.filter("semantic_similarity is not null")
                    .orderBy("semantic_similarity")
                    .limit(1)
                    .collect()
                )
                n_entities = db.read_entities(spark, video_id).select("entity_text").distinct().count()
            finally:
                spark.stop()
        except Exception:
            log.exception("Failed to load result stats")
            self.call_from_thread(
                self.query_one("#result-body", Static).update,
                "Processed, but stats couldn't be loaded -- see data/logs/tui.log",
            )
            return

        lines = [
            f"Segments: {agg['n']} · Topics: {agg['topics']} · Entities: {n_entities} unique",
            f"Avg compression ratio: {agg['avg_cr']:.2f} · Avg semantic similarity: {agg['avg_sim']:.2f}"
            if agg["avg_cr"] is not None and agg["avg_sim"] is not None
            else "",
        ]
        if divergent:
            d = divergent[0]
            lines.append(f"\nMost divergent segment (similarity: {d['semantic_similarity']:.2f})")
            lines.append(f"Extractive:  {(d['ext_summary'] or '')[:70]}")
            lines.append(f"Abstractive: {(d['abs_summary'] or '')[:70]}")

        self.call_from_thread(self.query_one("#result-body", Static).update, "\n".join(lines))

    async def on_pipeline_failed(self, message: PipelineFailed) -> None:
        self._stop_timer()
        try:
            await self.query_one("#progress-panel").remove()
        except Exception:
            pass
        shell = self.query_one("#shell", Vertical)
        await shell.mount(
            Vertical(
                Static("✗ Something went wrong", classes="error"),
                Static("Check data/logs/tui.log for details.", classes="muted"),
                Horizontal(
                    Button("Retry", id="btn-retry"),
                    Button("Quit", id="btn-quit"),
                ),
                id="error-panel",
            )
        )
        log.error("Pipeline failed: %s", message.reason)

    async def _reset_to_idle(self) -> None:
        self._stop_timer()
        for panel_id in ("#progress-panel", "#result-panel", "#error-panel"):
            try:
                await self.query_one(panel_id).remove()
            except Exception:
                pass
        shell = self.query_one("#shell", Vertical)
        try:
            self.query_one("#idle-panel")
        except Exception:
            await shell.mount(
                Vertical(
                    Static("Paste a YouTube URL below, then press Enter", classes="muted"),
                    Vertical(
                        Input(placeholder="https://www.youtube.com/watch?v=...", id="url-input"),
                        Static("", id="input-error", classes="error"),
                        id="input-row",
                    ),
                    Horizontal(
                        Button("Analyze", id="btn-analyze", variant="primary"),
                        Button("History", id="btn-history"),
                        Button("Quit", id="btn-quit"),
                        id="buttons",
                    ),
                    id="idle-panel",
                )
            )
        self.query_one("#url-input", Input).focus()


def main() -> None:
    PodscoreApp().run()


if __name__ == "__main__":
    main()
