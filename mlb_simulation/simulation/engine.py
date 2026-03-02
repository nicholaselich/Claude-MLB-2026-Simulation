"""Game simulation engine using a negative-binomial run-distribution model.

The public interface is intentionally minimal so the data layer and the
simulation logic stay decoupled.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import pandas as pd

from mlb_simulation.data.models import GameResult, Schedule, TeamProfile, TeamStats
from mlb_simulation.strength.team_model import HOME_ADV_FACTOR, LEAGUE_AVG_RPG, game_defense_rpg

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Drives a full or partial season simulation.

    Parameters
    ----------
    schedule:
        ``Schedule`` object for the season being simulated.
    team_stats:
        ``TeamStats`` object with pre-loaded hitting/pitching data.
    team_profiles:
        Dict mapping ``team_id`` → ``TeamProfile``.  When *None* or a team
        is missing, both clubs are treated as league-average.
    random_seed:
        Seed passed to both ``random`` and ``numpy.random`` for
        reproducibility.  ``None`` means non-deterministic runs.
    """

    def __init__(
        self,
        schedule: Schedule,
        team_stats: TeamStats,
        team_profiles: Optional[dict[int, TeamProfile]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.schedule = schedule
        self.team_stats = team_stats
        self.team_profiles = team_profiles or {}
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self._results: list[GameResult] = []
        self._rotation_idx: dict[int, int] = {}  # team_id → games started so far

    # ------------------------------------------------------------------
    # Core simulation method
    # ------------------------------------------------------------------

    def simulate_game(self, game_row: pd.Series) -> GameResult:
        """Simulate a single game and return a ``GameResult``.

        Uses a negative-binomial distribution (r=4) which closely matches
        the empirical MLB run distribution.  Teams without projection data
        are treated as league-average.

        Parameters
        ----------
        game_row:
            One row from the schedule DataFrame (must contain ``game_id``,
            ``home_team_id``, ``away_team_id``).
        """
        home_id = int(game_row["home_team_id"])
        away_id = int(game_row["away_team_id"])
        home_prof = self.team_profiles.get(home_id)
        away_prof = self.team_profiles.get(away_id)

        # Interaction model: team offense vs opponent defense, scaled to league avg
        if home_prof and away_prof:
            park_factor = home_prof.park_factor
            home_slot = self._rotation_idx.get(home_id, 0)
            away_slot = self._rotation_idx.get(away_id, 0)
            home_def_rpg = game_defense_rpg(home_prof, home_slot)
            away_def_rpg = game_defense_rpg(away_prof, away_slot)
            home_exp = (
                home_prof.offense_rpg
                * (away_def_rpg / LEAGUE_AVG_RPG)
                * HOME_ADV_FACTOR
                * park_factor
            )
            away_exp = (
                away_prof.offense_rpg
                * (home_def_rpg / LEAGUE_AVG_RPG)
                * park_factor
            )
            self._rotation_idx[home_id] = home_slot + 1
            self._rotation_idx[away_id] = away_slot + 1
        else:
            home_exp = away_exp = LEAGUE_AVG_RPG

        # Negative binomial: r=4 gives realistic run-distribution variance
        r = 4
        home_score = int(np.random.negative_binomial(r, r / (r + home_exp)))
        away_score = int(np.random.negative_binomial(r, r / (r + away_exp)))

        # Extra innings: play until untied (~0.5 runs/half-inning mean)
        innings = 9
        while home_score == away_score:
            innings += 1
            home_score += int(np.random.negative_binomial(r, r / (r + 0.5)))
            away_score += int(np.random.negative_binomial(r, r / (r + 0.5)))
            if innings > 30:   # safety valve
                home_score += 1
                break

        return GameResult(
            game_id=int(game_row["game_id"]),
            home_team_id=home_id,
            away_team_id=away_id,
            home_score=home_score,
            away_score=away_score,
            innings=innings,
        )

    # ------------------------------------------------------------------
    # Season-level helpers
    # ------------------------------------------------------------------

    def run_season(self) -> pd.DataFrame:
        """Simulate every unplayed game in the schedule.

        Returns
        -------
        DataFrame with one row per simulated game (columns match
        ``GameResult.to_series()``).
        """
        games = self.schedule.unplayed_games()
        logger.info("Simulating %d games …", len(games))

        for _, game_row in games.iterrows():
            result = self.simulate_game(game_row)
            self._results.append(result)

        return self.results_df()

    def results_df(self) -> pd.DataFrame:
        """Return all accumulated results as a DataFrame."""
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_series() for r in self._results])

    def standings(self) -> pd.DataFrame:
        """Compute win/loss standings from simulated results.

        Returns
        -------
        DataFrame with columns: team_id, wins, losses, win_pct
        sorted by win_pct descending.
        """
        df = self.results_df()
        if df.empty:
            return pd.DataFrame(columns=["team_id", "wins", "losses", "win_pct"])

        wins = df.groupby("winner_id").size().rename("wins")
        games_played = df.groupby("home_team_id").size().add(
            df.groupby("away_team_id").size(), fill_value=0
        )

        all_teams = pd.concat(
            [df["home_team_id"], df["away_team_id"]]
        ).unique()

        standings = pd.DataFrame({"team_id": all_teams}).set_index("team_id")
        standings["wins"] = wins
        standings["wins"] = standings["wins"].fillna(0).astype(int)
        standings["losses"] = (
            games_played.reindex(standings.index).fillna(0).astype(int)
            - standings["wins"]
        )
        standings["win_pct"] = standings["wins"] / (
            standings["wins"] + standings["losses"]
        ).replace(0, pd.NA)
        return (
            standings.reset_index()
            .sort_values("win_pct", ascending=False)
            .reset_index(drop=True)
        )
