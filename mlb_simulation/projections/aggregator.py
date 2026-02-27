"""Roll player-level Steamer projections up to team-level aggregates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TeamProjections:
    """Team-level aggregate projections keyed by MLB abbreviation."""

    abbrev: str
    weighted_wrc_plus: float
    starter_fip: float
    bullpen_fip: float
    starter_ip: float
    bullpen_ip: float


class ProjectionsAggregator:
    """Aggregate player-level Steamer projections to team level.

    Parameters
    ----------
    batting_df:
        Output of ``ProjectionsLoader.load_batting()``.
    pitching_df:
        Output of ``ProjectionsLoader.load_pitching()``.
    """

    def __init__(self, batting_df: pd.DataFrame, pitching_df: pd.DataFrame) -> None:
        self._batting = batting_df.copy()
        self._pitching = pitching_df.copy()

    def build(self) -> dict[str, TeamProjections]:
        """Build team-level projections keyed by MLB abbreviation."""
        result: dict[str, TeamProjections] = {}

        # ── Offense: PA-weighted wRC+ per team ────────────────────────
        bat = self._batting[self._batting["PA"] > 0].copy()
        wrc_by_team: dict[str, float] = {}
        for team, grp in bat.groupby("Team"):
            total_pa = grp["PA"].sum()
            if total_pa > 0:
                wrc_by_team[team] = float((grp["wRC+"] * grp["PA"]).sum() / total_pa)

        # ── Pitching: separate starters (GS > 0) and bullpen (GS == 0) ─
        pitch = self._pitching[self._pitching["IP"] > 0].copy()
        starters = pitch[pitch["GS"] > 0]
        bullpen = pitch[pitch["GS"] == 0]

        def _ip_weighted_fip(grp: pd.DataFrame) -> tuple[float, float]:
            total_ip = grp["IP"].sum()
            if total_ip == 0:
                return 4.20, 0.0
            fip = float((grp["FIP"] * grp["IP"]).sum() / total_ip)
            return fip, float(total_ip)

        starter_by_team: dict[str, tuple[float, float]] = {}
        for team, grp in starters.groupby("Team"):
            starter_by_team[team] = _ip_weighted_fip(grp)

        bullpen_by_team: dict[str, tuple[float, float]] = {}
        for team, grp in bullpen.groupby("Team"):
            bullpen_by_team[team] = _ip_weighted_fip(grp)

        all_teams = set(wrc_by_team) | set(starter_by_team) | set(bullpen_by_team)

        for team in sorted(all_teams):
            wrc_plus = wrc_by_team.get(team, 100.0)
            sp_fip, sp_ip = starter_by_team.get(team, (4.20, 0.0))
            bp_fip, bp_ip = bullpen_by_team.get(team, (4.20, 0.0))

            result[team] = TeamProjections(
                abbrev=team,
                weighted_wrc_plus=wrc_plus,
                starter_fip=sp_fip,
                bullpen_fip=bp_fip,
                starter_ip=sp_ip,
                bullpen_ip=bp_ip,
            )

        return result
