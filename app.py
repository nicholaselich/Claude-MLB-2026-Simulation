"""Flask web dashboard for the 2026 MLB Simulation."""

from __future__ import annotations

import json
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
    build_division_race_json,
    build_matchup_heatmap,
    build_playoff_bracket_html,
    build_scatter_chart,
    build_win_range_chart,
    build_ws_bar_chart,
)
from mlb_simulation.projections import ProjectionsAggregator, ProjectionsLoader
from mlb_simulation.simulation import MonteCarloSeasonSimulator, SimulationEngine
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
    "team_rosters": None,
    "sample_results": None,
    "team_profiles": None,
    "last_run": None,
    "error": None,
}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Roster builder
# ---------------------------------------------------------------------------

def _build_team_rosters(batting_df, pitching_df, teams_df) -> dict:
    """Return {team_name: {batters, starters, relievers}} from projection data."""
    abbrev_to_name = dict(zip(teams_df["abbreviation"], teams_df["name"]))
    rosters = {}
    for abbrev, name in abbrev_to_name.items():
        bdf = batting_df[batting_df["Team"] == abbrev].copy()
        top_batters = (
            bdf[bdf["PA"] >= 100]
            .sort_values("wRC+", ascending=False)
            .head(5)[["PlayerName", "PA", "wRC+"]]
        )
        top_batters = [
            {"PlayerName": r["PlayerName"], "PA": int(r["PA"]), "wRC+": int(round(r["wRC+"]))}
            for _, r in top_batters.iterrows()
        ]

        pdf = pitching_df[pitching_df["Team"] == abbrev].copy()
        top_starters = (
            pdf[pdf["GS"] >= 10]
            .sort_values("FIP")
            .head(5)[["PlayerName", "GS", "IP", "ERA", "FIP"]]
        )
        top_starters = [
            {"PlayerName": r["PlayerName"], "GS": int(r["GS"]),
             "IP": round(float(r["IP"]), 1), "ERA": round(float(r["ERA"]), 2),
             "FIP": round(float(r["FIP"]), 2)}
            for _, r in top_starters.iterrows()
        ]

        top_relievers = (
            pdf[(pdf["GS"] < 10) & (pdf["IP"] >= 15)]
            .sort_values("FIP")
            .head(5)[["PlayerName", "IP", "ERA", "FIP"]]
        )
        top_relievers = [
            {"PlayerName": r["PlayerName"], "IP": round(float(r["IP"]), 1),
             "ERA": round(float(r["ERA"]), 2), "FIP": round(float(r["FIP"]), 2)}
            for _, r in top_relievers.iterrows()
        ]

        rosters[name] = {"batters": top_batters, "starters": top_starters, "relievers": top_relievers}
    return rosters


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

        # Sample season for division race chart
        sample_engine = SimulationEngine(
            schedule=schedule,
            team_stats=team_stats,
            team_profiles=team_profiles,
            random_seed=42,
        )
        sample_results = sample_engine.run_season()
        sample_results = sample_results.merge(
            schedule.df[["game_id", "game_date"]], on="game_id", how="left"
        )

        team_rosters = _build_team_rosters(batting_df, pitching_df, teams_df)

        with _cache_lock:
            _cache["mc"] = mc
            _cache["df"] = df
            _cache["team_rosters"] = team_rosters
            _cache["sample_results"] = sample_results
            _cache["team_profiles"] = team_profiles
            _cache["teams_df"] = teams_df
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
        team_rosters = _cache["team_rosters"]
        sample_results = _cache["sample_results"]
        team_profiles = _cache.get("team_profiles")
        teams_df_cached = _cache.get("teams_df")
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

    # Scatter chart
    profiles_data = []
    if team_profiles and teams_df_cached is not None:
        for tid, profile in team_profiles.items():
            row = teams_df_cached[teams_df_cached["team_id"] == tid]
            if row.empty:
                continue
            name = row.iloc[0]["name"]
            league_id = int(row.iloc[0]["league_id"])
            ws_row = df[df["team_name"] == name]
            ws_pct = float(ws_row["ws_win_pct"].iloc[0]) if not ws_row.empty else 0.0
            profiles_data.append({
                "team_name": name,
                "offense_rpg": profile.offense_rpg,
                "defense_rpg": profile.defense_rpg,
                "league_id": league_id,
                "ws_win_pct": ws_pct,
            })
    scatter_html = pio.to_html(build_scatter_chart(profiles_data), **_pio_opts) if profiles_data else ""

    # Division race — pass raw data as JSON; chart built client-side to avoid hidden-div sizing issue
    race_data_json = "{}"
    if sample_results is not None and teams_df_cached is not None:
        race_data_json = build_division_race_json(sample_results, teams_df_cached)

    # Playoff bracket HTML
    playoff_bracket_html = ""
    if mc is not None and teams_df_cached is not None:
        playoff_bracket_html = build_playoff_bracket_html(
            df, mc.wc_wins, mc.ds_wins, N_SIMULATIONS, mc._team_info
        )

    # Win distributions for histogram in modal
    win_dist_by_name = {}
    if mc is not None:
        for tid, wins in mc.win_distributions.items():
            name = mc._team_info.get(tid, {}).get("name", "")
            if name:
                win_dist_by_name[name] = [int(w) for w in wins]

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
        scatter_html=scatter_html,
        race_data_json=race_data_json,
        playoff_bracket_html=playoff_bracket_html,
        last_run=last_run_str,
        n_simulations=N_SIMULATIONS,
        team_rosters_json=json.dumps(team_rosters),
        win_distributions_json=json.dumps(win_dist_by_name),
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
