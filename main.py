"""Full 2026 MLB season simulation pipeline.

Steps
-----
1. Fetch 2026 schedule and team metadata from the MLB Stats API.
2. Load Steamer projections from data/hitters.json and data/pitchers.json.
3. Aggregate projections to team level and build TeamProfiles.
4. Run 1 000-iteration Monte Carlo season simulation.
5. Print divisional standings with playoff/WS probabilities.
6. Save visualisation charts.
"""

import logging
from pathlib import Path

from mlb_simulation.api import MLBClient
from mlb_simulation.data import Schedule, TeamStats
from mlb_simulation.output import build_standings_df, plot_ws_probabilities, print_standings
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
RANDOM_SEED = 42


def main() -> None:
    client = MLBClient()

    # ── 1. Fetch schedule and teams ───────────────────────────────────
    logger.info("Fetching 2026 schedule …")
    schedule_df = client.get_schedule(season=2026)
    schedule = Schedule(schedule_df)
    logger.info("Schedule: %s  (opening day: %s)", schedule, schedule_df["game_date"].min().date())

    logger.info("Fetching teams …")
    teams_df = client.get_teams()
    logger.info("Teams loaded: %d", len(teams_df))

    # ── 2. Load Steamer projections ───────────────────────────────────
    logger.info("Loading Steamer projections from %s …", DATA_DIR)
    loader = ProjectionsLoader(data_dir=DATA_DIR)
    batting_df = loader.load_batting()
    pitching_df = loader.load_pitching()
    logger.info(
        "Projections loaded: %d batters, %d pitchers",
        len(batting_df), len(pitching_df),
    )

    # ── 3. Aggregate projections → team profiles ──────────────────────
    logger.info("Aggregating projections …")
    aggregator = ProjectionsAggregator(batting_df, pitching_df)
    team_projections = aggregator.build()
    logger.info("Team projections built for %d teams.", len(team_projections))

    # Sanity check: wRC+ range
    wrc_values = [tp.weighted_wrc_plus for tp in team_projections.values()]
    logger.info(
        "wRC+ range: %.1f – %.1f  (expected 85–115)",
        min(wrc_values), max(wrc_values),
    )

    strength_model = TeamStrengthModel(team_projections, teams_df)
    team_profiles = strength_model.build_profiles()
    logger.info("TeamProfiles built for %d teams.", len(team_profiles))

    # Sanity check: RPG range
    off_rpg = [tp.offense_rpg for tp in team_profiles.values()]
    def_rpg = [tp.defense_rpg for tp in team_profiles.values()]
    logger.info(
        "Offense RPG range: %.2f – %.2f  (expected 3.8–5.2)",
        min(off_rpg), max(off_rpg),
    )
    logger.info(
        "Defense RPG range: %.2f – %.2f  (expected 3.8–5.2)",
        min(def_rpg), max(def_rpg),
    )

    # TeamStats is kept for engine compatibility (not used in the NB model)
    team_stats = TeamStats(
        hitting=batting_df.rename(columns={"Team": "team_name"}),
        pitching=pitching_df.rename(columns={"Team": "team_name"}),
    )

    # ── 4. Monte Carlo simulation ─────────────────────────────────────
    logger.info(
        "Running %d Monte Carlo simulations (seed=%s) …",
        N_SIMULATIONS, RANDOM_SEED,
    )
    mc = MonteCarloSeasonSimulator(
        schedule=schedule,
        team_stats=team_stats,
        team_profiles=team_profiles,
        teams_df=teams_df,
        n_simulations=N_SIMULATIONS,
        random_seed=RANDOM_SEED,
    )
    probabilities = mc.run()

    # ── 5. Print standings ────────────────────────────────────────────
    print_standings(probabilities)

    # Machine-readable version
    standings_df = build_standings_df(probabilities)
    csv_path = Path("simulation_results.csv")
    standings_df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    # ── 6. Charts ─────────────────────────────────────────────────────
    plot_ws_probabilities(probabilities, output_path="ws_probabilities.png")

    # Division-race chart: run one representative single-season simulation
    logger.info("Generating division race chart (one simulation) …")
    sample_engine = SimulationEngine(
        schedule=schedule,
        team_stats=team_stats,
        team_profiles=team_profiles,
        random_seed=RANDOM_SEED,
    )
    sample_results = sample_engine.run_season()

    # AL East division_id = 201 (standard MLB Stats API value)
    from mlb_simulation.output import plot_division_race
    al_east_id = int(
        teams_df[teams_df["division_name"].str.contains("East", na=False)
                 & (teams_df["league_id"] == 103)]["division_id"].iloc[0]
    )
    plot_division_race(
        sample_results, teams_df, al_east_id,
        output_path="al_east_race.png",
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
