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
    rotation_fips: list  # individual starter FIPs sorted best→worst (FIP ascending, GS≥10)


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

        def _leverage_adjusted_fip(grp: pd.DataFrame) -> tuple[float, float]:
            """Leverage-adjusted bullpen FIP: best arms get higher weight.

            Closer (best FIP, IP≥15): 1.0 IP weight
            Setup x2 (next two by FIP, IP≥15): 0.5 IP weight each
            Middle/mop-up: IP-weighted for remaining 1.5 of 3.5 total IP

            Upweighting the top arms models that closers and setup men face
            batters in high-leverage situations, not just any inning.
            """
            total_ip = float(grp["IP"].sum())
            qual = grp[grp["IP"] >= 15].sort_values("FIP").reset_index(drop=True)
            if len(qual) == 0:
                return _ip_weighted_fip(grp)

            lev_weights = [1.0, 0.5, 0.5]   # closer, setup1, setup2
            n = min(len(qual), len(lev_weights))
            weighted = sum(qual.iloc[i]["FIP"] * lev_weights[i] for i in range(n))
            assigned = sum(lev_weights[:n])
            remaining = 3.5 - assigned

            if len(qual) > n:
                rest = qual.iloc[n:]
                rest_fip = float((rest["FIP"] * rest["IP"]).sum() / rest["IP"].sum())
                weighted += rest_fip * remaining
            else:
                weighted += qual.iloc[n - 1]["FIP"] * remaining

            return weighted / 3.5, total_ip

        starter_by_team: dict[str, tuple[float, float]] = {}
        for team, grp in starters.groupby("Team"):
            starter_by_team[team] = _ip_weighted_fip(grp)

        bullpen_by_team: dict[str, tuple[float, float]] = {}
        for team, grp in bullpen.groupby("Team"):
            bullpen_by_team[team] = _leverage_adjusted_fip(grp)

        # Rotation: starters with GS >= 10, sorted by FIP ascending (best first)
        rotation_by_team: dict[str, list] = {}
        for team, grp in starters[starters["GS"] >= 10].groupby("Team"):
            rotation_by_team[team] = grp.sort_values("FIP")["FIP"].tolist()

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
                rotation_fips=rotation_by_team.get(team, []),
            )

        return result
