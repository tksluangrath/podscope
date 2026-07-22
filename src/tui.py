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
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Input, ProgressBar, Static

# ingest.py imports only yt_dlp -- no Spark/spaCy/Ollama -- ~0.1s, well
# within the <2s cold-start budget, so it's safe at module level (unlike
# src.run/src.db, which stay import-inside-handler-only).
from src import ingest

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
        # ingest.is_playlist_url is the one definition of "what counts as a
        # playlist URL" -- src/run.py's CLI dispatch uses the same rule.
        return True, "playlist" if ingest.is_playlist_url(url) else "video"
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


class EpisodeDone(Message):
    """One PODSCOPE_EPISODE_DONE sentinel line, parsed from run.py's stdout
    by run_pipeline (see EPISODE_DONE_RE below)."""

    def __init__(self, index: int, total: int, video_id: str, ok: bool) -> None:
        self.index = index
        self.total = total
        self.video_id = video_id
        self.ok = ok
        super().__init__()


class PlaylistComplete(Message):
    def __init__(self, url: str, succeeded_video_ids: list[str], total: int, failed: int) -> None:
        self.url = url
        self.succeeded_video_ids = succeeded_video_ids
        self.total = total
        self.failed = failed
        super().__init__()


EPISODE_DONE_RE = re.compile(r"PODSCOPE_EPISODE_DONE (\d+)/(\d+) (\S+) (ok|failed)")


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
        table = self.query_one("#history-table", DataTable)
        table.cursor_type = "row"
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
                # row_key = video_id, so on_data_table_row_selected doesn't
                # need to re-parse it back out of the displayed column text.
                table.add_row(
                    r["title"] or "(untitled)", str(r["processed_at"]), r["video_id"], key=r["video_id"]
                )
            status.update(
                f"{len(rows)} video(s) processed -- press enter for detail" if rows else "No videos processed yet"
            )

        self.app.call_from_thread(render)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        video_id = str(event.row_key.value)
        row = self.query_one("#history-table", DataTable).get_row(event.row_key)
        title = row[0]
        self.app.push_screen(VideoDetailScreen(video_id, title))


class VideoDetailScreen(Screen):
    """State 8 -- per-video detail: segment/entity stats, top entities, sample summaries."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("q", "app.pop_screen", "back"),
    ]

    def __init__(self, video_id: str, title: str) -> None:
        super().__init__()
        self.video_id = video_id
        self.title = title

    def compose(self) -> ComposeResult:
        yield Static(self.title, classes="accent")
        with VerticalScroll(id="detail-body"):
            yield Static("Loading...", id="detail-status", classes="muted")
        yield Footer()

    def on_mount(self) -> None:
        self.load_detail()

    @work(thread=True)
    def load_detail(self) -> None:
        # Imported here, not at module level -- avoids JVM init on TUI launch.
        from src import db

        status = self.query_one("#detail-status", Static)
        try:
            spark = db.build_spark("data/iceberg")
            try:
                segs = db.read_segments(spark, self.video_id)
                agg = segs.selectExpr(
                    "count(*) as n",
                    "count(distinct topic_label) as topics",
                    "avg(compression_ratio) as avg_cr",
                    "avg(semantic_similarity) as avg_sim",
                ).collect()[0]
                sample = segs.orderBy("segment_id").limit(5).collect()
                ents = db.read_entities(spark, self.video_id)
                top_entities = (
                    ents.groupBy("entity_text", "entity_type")
                    .count()
                    .orderBy("count", ascending=False)
                    .limit(10)
                    .collect()
                )
            finally:
                spark.stop()
        except Exception:
            log.exception("Failed to load video detail")
            self.app.call_from_thread(status.update, "Could not load detail -- see data/logs/tui.log")
            return

        lines = [
            f"Segments: {agg['n']} · Topics: {agg['topics']}",
            f"Avg compression ratio: {agg['avg_cr']:.2f} · Avg semantic similarity: {agg['avg_sim']:.2f}"
            if agg["avg_cr"] is not None and agg["avg_sim"] is not None
            else "",
            "",
            "Top entities:",
        ]
        lines += [f"  {e['entity_text']} ({e['entity_type']}): {e['count']}" for e in top_entities] or ["  none"]
        lines += ["", "Sample segments:"]
        for s in sample:
            lines.append(f"  [{s['segment_id']}] {(s['ext_summary'] or '')[:100]}")

        self.app.call_from_thread(status.update, "\n".join(lines))


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
    #episode-rows {
        height: auto;
        max-height: 8;
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

        if self._url_kind == "playlist":
            panel = Vertical(
                Static(f'Playlist: "{url}"'),
                Static("", id="step-line", classes="accent"),
                VerticalScroll(id="episode-rows"),
                ProgressBar(id="progress-bar", show_eta=False),
                Horizontal(Button("Cancel", id="btn-cancel")),
                id="progress-panel",
            )
        else:
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
        self.run_pipeline(url, self._url_kind)

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
    def run_pipeline(self, url: str, kind: str) -> None:
        # Imported here, not at module level -- subprocess isolates the
        # Spark JVM / Ollama calls in a child process so the TUI's own
        # event loop never blocks on their ~8s startup cost. --url auto-
        # detects playlist vs. video (same rule as validate_url), so no
        # separate CLI flag/subprocess branch is needed here either.
        import subprocess
        import sys

        def peek_video_id() -> str | None:
            try:
                return ingest.peek_video_id(url)
            except Exception:
                log.exception("peek_video_id failed for %s", url)
                return None

        succeeded: list[str] = []
        episode_total = 0
        episode_failed = 0
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
                m = EPISODE_DONE_RE.search(line)
                if m:
                    index, total, video_id, status = m.groups()
                    episode_total = int(total)
                    ok = status == "ok"
                    if ok:
                        succeeded.append(video_id)
                    else:
                        episode_failed += 1
                    self.post_message(EpisodeDone(int(index), episode_total, video_id, ok))
                else:
                    self.post_message(PipelineProgress(line))
            process.wait()
            if process.returncode == 0:
                if kind == "playlist":
                    self.post_message(PlaylistComplete(url, succeeded, episode_total, episode_failed))
                else:
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

    async def on_episode_done(self, message: EpisodeDone) -> None:
        # Episodes are processed strictly in order by process_playlist, one
        # at a time -- so the row right after the one this sentinel just
        # completed is always the one now running. No separate "started"
        # sentinel needed for that inference.
        rows = self.query_one("#episode-rows", VerticalScroll)
        if not rows.children:
            await rows.mount_all(
                Static(f"○ Episode {i}/{message.total} -- waiting", id=f"episode-row-{i}", classes="muted")
                for i in range(1, message.total + 1)
            )

        row = self.query_one(f"#episode-row-{message.index}", Static)
        glyph, cls, label = ("✓", "success", "done") if message.ok else ("✗", "error", "failed")
        row.set_classes(cls)
        row.update(f"{glyph} Episode {message.index}/{message.total} ({message.video_id}) -- {label}")

        next_index = message.index + 1
        if next_index <= message.total:
            try:
                next_row = self.query_one(f"#episode-row-{next_index}", Static)
                next_row.set_classes("accent")
                next_row.update(f"⠋ Episode {next_index}/{message.total} -- running")
            except Exception:
                pass

        self.query_one("#progress-bar", ProgressBar).update(total=message.total, progress=message.index)
        self._phase = f"{message.index}/{message.total} episodes processed"
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

    async def on_playlist_complete(self, message: PlaylistComplete) -> None:
        self._stop_timer()
        await self.query_one("#progress-panel").remove()
        shell = self.query_one("#shell", Vertical)
        succeeded = len(message.succeeded_video_ids)
        await shell.mount(
            Vertical(
                Static(
                    f"✓ Complete — {succeeded}/{message.total} episodes succeeded "
                    f"({message.failed} failed)",
                    classes="success",
                ),
                Static("", id="result-body", classes="muted"),
                Horizontal(
                    Button("Analyze Another", id="btn-analyze-another"),
                    Button("View in History", id="btn-history"),
                    Button("Quit", id="btn-quit"),
                ),
                id="result-panel",
            )
        )
        if message.succeeded_video_ids:
            self.load_result_stats(message.succeeded_video_ids)
        else:
            self.query_one("#result-body", Static).update(
                "No episodes completed successfully -- see data/logs/tui.log."
            )

    @work(thread=True)
    def load_result_stats(self, video_ids: str | list[str]) -> None:
        # Accepts either one video_id (single-video run) or a list (playlist
        # run, aggregated across every episode that succeeded) -- one query
        # shape for both instead of a parallel single-video-only path.
        from src import db

        ids = [video_ids] if isinstance(video_ids, str) else video_ids

        try:
            spark = db.build_spark("data/iceberg")
            try:
                all_segs = db.read_segments(spark)
                segs = all_segs.filter(all_segs.video_id.isin(ids))
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
                all_ents = db.read_entities(spark)
                n_entities = (
                    all_ents.filter(all_ents.video_id.isin(ids))
                    .select("entity_text")
                    .distinct()
                    .count()
                )
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
