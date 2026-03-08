"""Flask web dashboard for the 2026 MLB Simulation."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

import plotly.io as pio
from flask import Flask, redirect, render_template, url_for

from mlb_simulation.api import MLBClient
from mlb_simulation.data import Schedule, TeamStats
from mlb_simulation.output import build_standings_df
from mlb_simulation.output.web import (
    build_division_tables_html,
    build_matchup_heatmap,
    build_win_range_chart,
    build_ws_bar_chart,
)
from mlb_simulation.projections import ProjectionsAggregator, ProjectionsLoader
from mlb_simulation.simulation import MonteCarloSeasonSimulator
from mlb_simulation.strength import TeamStrengthModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
N_SIMULATIONS = 1000

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ---------------------------------------------------------------------------
# Shared cache
# ---------------------------------------------------------------------------
_cache: dict = {
    "status": "idle",   # "idle" | "running" | "ready" | "error"
    "mc": None,
    "df": None,
    "last_run": None,
    "error": None,
}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Background simulation
# ---------------------------------------------------------------------------

def _run_simulation() -> None:
    with _cache_lock:
        if _cache["status"] == "running":
            return
        _cache["status"] = "running"
        _cache["error"] = None

    try:
        logger.info("Dashboard: starting simulation …")
        client = MLBClient()

        schedule_df = client.get_schedule(season=2026)
        schedule = Schedule(schedule_df)

        teams_df = client.get_teams()

        loader = ProjectionsLoader(data_dir=DATA_DIR)
        batting_df = loader.load_batting()
        pitching_df = loader.load_pitching()

        team_projections = ProjectionsAggregator(batting_df, pitching_df).build()

        team_stats = TeamStats(
            hitting=batting_df.rename(columns={"Team": "team_name"}),
            pitching=pitching_df.rename(columns={"Team": "team_name"}),
        )

        team_profiles = TeamStrengthModel(team_projections, teams_df).build_profiles()

        mc = MonteCarloSeasonSimulator(
            schedule=schedule,
            team_stats=team_stats,
            team_profiles=team_profiles,
            teams_df=teams_df,
            n_simulations=N_SIMULATIONS,
            random_seed=42,
        )
        probabilities = mc.run()
        df = build_standings_df(probabilities)

        with _cache_lock:
            _cache["mc"] = mc
            _cache["df"] = df
            _cache["status"] = "ready"
            _cache["last_run"] = datetime.now()

        logger.info("Dashboard: simulation complete.")

    except Exception as exc:  # noqa: BLE001
        logger.exception("Dashboard: simulation failed.")
        with _cache_lock:
            _cache["status"] = "error"
            _cache["error"] = str(exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    with _cache_lock:
        status = _cache["status"]
        mc = _cache["mc"]
        df = _cache["df"]
        last_run = _cache["last_run"]
        error = _cache["error"]

    if status in ("idle", "running"):
        return render_template("index.html", loading=True, status=status)

    if status == "error":
        return render_template("index.html", loading=False, status=status, error=error)

    # Build Plotly HTML snippets (no inline JS bundle — CDN used in template)
    _pio_opts = dict(full_html=False, include_plotlyjs=False)

    ws_bar_html = pio.to_html(build_ws_bar_chart(df), **_pio_opts)
    win_range_html = pio.to_html(build_win_range_chart(df), **_pio_opts)
    heatmap_html = pio.to_html(
        build_matchup_heatmap(mc.ws_matchup_counts, mc._team_info, N_SIMULATIONS),
        **_pio_opts,
    )
    al_division_html, nl_division_html = build_division_tables_html(df)

    last_run_str = last_run.strftime("%Y-%m-%d %H:%M:%S") if last_run else "—"

    return render_template(
        "index.html",
        loading=False,
        status=status,
        ws_bar_html=ws_bar_html,
        win_range_html=win_range_html,
        heatmap_html=heatmap_html,
        al_division_html=al_division_html,
        nl_division_html=nl_division_html,
        last_run=last_run_str,
        n_simulations=N_SIMULATIONS,
    )


@app.route("/rerun", methods=["POST"])
def rerun():
    with _cache_lock:
        status = _cache["status"]

    if status != "running":
        threading.Thread(target=_run_simulation, daemon=True).start()

    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

threading.Thread(target=_run_simulation, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=False, port=5001)
