"""Monte Carlo season simulator: run N full seasons and aggregate probabilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlb_simulation.data.models import Schedule, TeamProfile, TeamStats
from mlb_simulation.simulation.engine import SimulationEngine
from mlb_simulation.simulation.playoffs import PlayoffSimulator

logger = logging.getLogger(__name__)

# MLB league IDs in the Stats API
_AL_ID = 103
_NL_ID = 104


@dataclass
class SeasonProbabilities:
    """Per-team aggregated statistics across N Monte Carlo simulations."""

    team_id: int
    team_name: str
    division_id: int
    division_name: str
    league_id: int
    league_name: str
    avg_wins: float = 0.0
    avg_losses: float = 0.0
    p10_wins: float = 0.0         # 10th-percentile wins (floor of likely range)
    p90_wins: float = 0.0         # 90th-percentile wins (ceiling of likely range)
    div_title_pct: float = 0.0    # fraction of sims where team won its division
    playoff_pct: float = 0.0      # fraction of sims where team made playoffs
    ws_win_pct: float = 0.0       # fraction of sims where team won World Series


class MonteCarloSeasonSimulator:
    """Run N full-season simulations and aggregate per-team probabilities.

    Parameters
    ----------
    schedule:
        ``Schedule`` for the season to simulate (2026).
    team_stats:
        ``TeamStats`` object (used by ``SimulationEngine``).
    team_profiles:
        Dict mapping ``team_id`` → ``TeamProfile``.
    teams_df:
        DataFrame from ``MLBClient.get_teams()`` with division/league info.
    n_simulations:
        Number of Monte Carlo iterations.
    random_seed:
        Base seed; each simulation uses ``random_seed + i``.
    """

    def __init__(
        self,
        schedule: Schedule,
        team_stats: TeamStats,
        team_profiles: dict[int, TeamProfile],
        teams_df: pd.DataFrame,
        n_simulations: int = 1000,
        random_seed: int | None = None,
    ) -> None:
        self.schedule = schedule
        self.team_stats = team_stats
        self.team_profiles = team_profiles
        self.teams_df = teams_df.copy()
        self.n_simulations = n_simulations
        self.random_seed = random_seed

        # Pre-build a lookup: team_id → row with division/league info
        self._team_info: dict[int, dict] = {
            int(r["team_id"]): r.to_dict()
            for _, r in self.teams_df.iterrows()
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> list[SeasonProbabilities]:
        """Execute all simulations and return per-team probabilities."""
        # Accumulation counters
        all_wins: dict[int, list] = {tid: [] for tid in self._team_info}
        div_titles: dict[int, int] = {tid: 0 for tid in self._team_info}
        playoff_apps: dict[int, int] = {tid: 0 for tid in self._team_info}
        ws_wins: dict[int, int] = {tid: 0 for tid in self._team_info}
        ws_matchup_counts: dict[tuple, int] = {}
        wc_wins: dict[int, int] = {tid: 0 for tid in self._team_info}
        ds_wins: dict[int, int] = {tid: 0 for tid in self._team_info}

        logger.info("Starting %d Monte Carlo simulations …", self.n_simulations)

        for i in range(self.n_simulations):
            if (i + 1) % 100 == 0:
                logger.info("  Simulation %d / %d …", i + 1, self.n_simulations)

            seed = (self.random_seed + i) if self.random_seed is not None else None
            standings_df, ws_winner_id, al_champ_id, nl_champ_id, bracket = self._run_one_simulation(seed)

            # Track per-sim wins for confidence intervals
            for _, row in standings_df.iterrows():
                tid = int(row["team_id"])
                if tid in all_wins:
                    all_wins[tid].append(float(row.get("wins", 0)))

            # Track WS matchups
            matchup = (al_champ_id, nl_champ_id)
            ws_matchup_counts[matchup] = ws_matchup_counts.get(matchup, 0) + 1

            # Track per-round wins
            for wc_winner in bracket["AL"]["wild_card"] + bracket["NL"]["wild_card"]:
                if wc_winner in wc_wins:
                    wc_wins[wc_winner] += 1
            for ds_winner in bracket["AL"]["division_series"] + bracket["NL"]["division_series"]:
                if ds_winner in ds_wins:
                    ds_wins[ds_winner] += 1

            # Determine division winners + wild cards per league
            al_seeds, nl_seeds = self._get_playoff_seeds(standings_df)

            all_playoff_ids = set(al_seeds) | set(nl_seeds)

            # Division titles: seeds 1-3 within each league (division winners)
            # Determine by picking highest win_pct team per division
            for league_seeds in (al_seeds, nl_seeds):
                div_winner_ids = self._division_winners_from_seeds(
                    league_seeds, standings_df
                )
                for did in div_winner_ids:
                    if did in div_titles:
                        div_titles[did] += 1

            for pid in all_playoff_ids:
                if pid in playoff_apps:
                    playoff_apps[pid] += 1

            if ws_winner_id in ws_wins:
                ws_wins[ws_winner_id] += 1

        n = self.n_simulations
        self.ws_matchup_counts = ws_matchup_counts
        self.win_distributions = all_wins
        self.wc_wins = wc_wins
        self.ds_wins = ds_wins
        results: list[SeasonProbabilities] = []
        for team_id, info in self._team_info.items():
            wins_list = all_wins[team_id]
            avg_w = float(np.mean(wins_list)) if wins_list else 0.0
            sp = SeasonProbabilities(
                team_id=team_id,
                team_name=str(info.get("name", "")),
                division_id=int(info.get("division_id", 0) or 0),
                division_name=str(info.get("division_name", "")),
                league_id=int(info.get("league_id", 0) or 0),
                league_name=str(info.get("league_name", "")),
                avg_wins=avg_w,
                avg_losses=162.0 - avg_w,
                p10_wins=float(np.percentile(wins_list, 10)) if wins_list else 0.0,
                p90_wins=float(np.percentile(wins_list, 90)) if wins_list else 0.0,
                div_title_pct=div_titles[team_id] / n * 100,
                playoff_pct=playoff_apps[team_id] / n * 100,
                ws_win_pct=ws_wins[team_id] / n * 100,
            )
            results.append(sp)

        results.sort(key=lambda s: s.ws_win_pct, reverse=True)
        logger.info("Monte Carlo complete.")
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def top_matchups(self, n: int = 8) -> list[tuple[str, str, float]]:
        """Return the top N most frequent World Series matchups.

        Returns a list of (al_team_name, nl_team_name, pct) tuples, sorted
        by frequency descending. Call after ``run()``.
        """
        sorted_matchups = sorted(
            self.ws_matchup_counts.items(), key=lambda x: x[1], reverse=True
        )[:n]
        result = []
        for (al_id, nl_id), count in sorted_matchups:
            al_name = self._team_info.get(al_id, {}).get("name", f"Team {al_id}")
            nl_name = self._team_info.get(nl_id, {}).get("name", f"Team {nl_id}")
            result.append((al_name, nl_name, count / self.n_simulations * 100))
        return result

    def _run_one_simulation(
        self, seed: int | None
    ) -> tuple[pd.DataFrame, int, int, int]:
        """Run a single season simulation.

        Returns
        -------
        (standings_df, ws_winner_team_id, al_champ_team_id, nl_champ_team_id)
        """
        engine = SimulationEngine(
            schedule=self.schedule,
            team_stats=self.team_stats,
            team_profiles=self.team_profiles,
            random_seed=seed,
        )
        engine.run_season()
        standings_df = engine.standings()

        # Join standings with team metadata for division/league info
        standings_df = standings_df.merge(
            self.teams_df[["team_id", "name", "league_id", "league_name",
                           "division_id", "division_name"]],
            on="team_id",
            how="left",
        )

        al_seeds, nl_seeds = self._get_playoff_seeds(standings_df)
        playoff_sim = PlayoffSimulator(self.team_profiles)
        bracket = playoff_sim.simulate_bracket(al_seeds, nl_seeds)
        ws_winner_id = int(bracket["world_series_winner"])
        al_champ_id = int(bracket["AL"]["lcs_winner"])
        nl_champ_id = int(bracket["NL"]["lcs_winner"])

        return standings_df, ws_winner_id, al_champ_id, nl_champ_id, bracket

    def _get_playoff_seeds(
        self, standings_df: pd.DataFrame
    ) -> tuple[list[int], list[int]]:
        """Determine 6 playoff seeds per league.

        Seeds 1-3: division winners (ordered by win_pct descending).
        Seeds 4-6: best remaining teams by win_pct.

        Returns
        -------
        (al_seeds, nl_seeds) — each a list of 6 team IDs, best-record first.
        """
        def _seeds_for_league(league_id: int) -> list[int]:
            lg = standings_df[standings_df["league_id"] == league_id].copy()
            if lg.empty:
                return []

            # Division winners: best record in each division
            div_winners: list[int] = []
            for div_id in lg["division_id"].unique():
                div = lg[lg["division_id"] == div_id].sort_values(
                    "win_pct", ascending=False
                )
                if not div.empty:
                    div_winners.append(int(div.iloc[0]["team_id"]))

            # Wild cards: top 3 remaining by win_pct
            non_div = lg[~lg["team_id"].isin(div_winners)].sort_values(
                "win_pct", ascending=False
            )
            wild_cards = non_div["team_id"].head(3).tolist()

            # Seed all 6 by win_pct
            all_six = lg[lg["team_id"].isin(div_winners + wild_cards)].sort_values(
                "win_pct", ascending=False
            )
            return [int(t) for t in all_six["team_id"].tolist()[:6]]

        al_seeds = _seeds_for_league(_AL_ID)
        nl_seeds = _seeds_for_league(_NL_ID)

        # Pad with fallback team IDs if fewer than 6 found (edge cases)
        for seeds in (al_seeds, nl_seeds):
            while len(seeds) < 6:
                seeds.append(seeds[-1] if seeds else 0)

        return al_seeds, nl_seeds

    def _division_winners_from_seeds(
        self, league_seeds: list[int], standings_df: pd.DataFrame
    ) -> list[int]:
        """From a list of seeded team IDs, return the division winner for each division."""
        if standings_df.empty or "division_id" not in standings_df.columns:
            return []
        relevant = standings_df[standings_df["team_id"].isin(league_seeds)].copy()
        div_winners = []
        for div_id in relevant["division_id"].unique():
            div = relevant[relevant["division_id"] == div_id].sort_values(
                "win_pct", ascending=False
            )
            if not div.empty:
                div_winners.append(int(div.iloc[0]["team_id"]))
        return div_winners
