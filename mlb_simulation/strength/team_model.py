"""Convert team-level projections into runs-per-game strength profiles."""

from __future__ import annotations

import logging

import pandas as pd

from mlb_simulation.data.models import TeamProfile
from mlb_simulation.projections.aggregator import TeamProjections

logger = logging.getLogger(__name__)

# Module-level constants (overridable by callers)
LEAGUE_AVG_RPG: float = 4.50
LEAGUE_AVG_FIP: float = 4.20
HOME_ADV_FACTOR: float = 1.03
STARTER_IP: float = 5.5   # average innings per game from starters
BULLPEN_IP: float = 3.5   # average innings per game from bullpen

# 5-year park factors, FanGraphs 2025 (source: fangraphs.com/guts.aspx?type=pf&season=2025)
# Values >1.0 favour hitters; <1.0 favour pitchers; 1.0 = neutral.
# Both teams' expected run totals are multiplied by the home park factor.
PARK_FACTORS: dict[str, float] = {
    "ARI": 1.01,  # Chase Field
    "ATL": 1.00,  # Truist Park
    "BAL": 0.99,  # Camden Yards
    "BOS": 1.04,  # Fenway Park
    "CHC": 0.98,  # Wrigley Field
    "CIN": 1.05,  # Great American Ball Park
    "CLE": 0.99,  # Progressive Field
    "COL": 1.13,  # Coors Field
    "CWS": 1.00,  # Guaranteed Rate Field
    "DET": 1.00,  # Comerica Park
    "HOU": 0.99,  # Minute Maid Park
    "KC":  1.03,  # Kauffman Stadium
    "LAA": 1.01,  # Angel Stadium
    "LAD": 0.99,  # Dodger Stadium
    "MIA": 1.01,  # loanDepot park
    "MIL": 0.99,  # American Family Field
    "MIN": 1.01,  # Target Field
    "NYM": 0.96,  # Citi Field
    "NYY": 0.99,  # Yankee Stadium
    "ATH": 1.03,  # Sutter Health Park / future Las Vegas ballpark
    "PHI": 1.01,  # Citizens Bank Park
    "PIT": 1.02,  # PNC Park
    "SD":  0.96,  # Petco Park
    "SEA": 0.94,  # T-Mobile Park
    "SF":  0.97,  # Oracle Park
    "STL": 0.98,  # Busch Stadium
    "TB":  1.01,  # Tropicana Field
    "TEX": 0.99,  # Globe Life Field
    "TOR": 0.99,  # Rogers Centre
    "WSH": 1.00,  # Nationals Park
}


def game_defense_rpg(profile: TeamProfile, starter_slot: int = 0) -> float:
    """Compute per-game defense RPG using the team's rotation slot (0 = ace).

    Blends the slot's starter FIP with bullpen FIP, then scales to RPG.
    Falls back to the pre-blended ``defense_rpg`` if no rotation is stored.
    """
    if not profile.rotation:
        return profile.defense_rpg
    starter_fip = profile.rotation[starter_slot % len(profile.rotation)]
    blend = (starter_fip * STARTER_IP + profile.bullpen_fip * BULLPEN_IP) / 9.0
    return (blend / LEAGUE_AVG_FIP) * LEAGUE_AVG_RPG


class TeamStrengthModel:
    """Convert ``TeamProjections`` to ``TeamProfile`` objects.

    Parameters
    ----------
    team_projections:
        Dict keyed by MLB abbreviation, as returned by
        ``ProjectionsAggregator.build()``.
    teams_df:
        DataFrame from ``MLBClient.get_teams()`` with columns
        ``team_id, name, abbreviation``.
    """

    def __init__(
        self,
        team_projections: dict[str, TeamProjections],
        teams_df: pd.DataFrame,
    ) -> None:
        self._projections = team_projections
        self._teams_df = teams_df

    def build_profiles(self) -> dict[int, TeamProfile]:
        """Build a ``TeamProfile`` for every team in ``teams_df``.

        Teams without projection data receive league-average profiles and
        a warning is logged.

        Returns
        -------
        Dict keyed by ``team_id`` (int).
        """
        profiles: dict[int, TeamProfile] = {}

        for _, row in self._teams_df.iterrows():
            team_id = int(row["team_id"])
            team_name = str(row["name"])
            abbrev = str(row["abbreviation"])
            proj = self._projections.get(abbrev)

            if proj is None:
                logger.warning(
                    "No projection data for %s (%s); using league-average profile.",
                    team_name,
                    abbrev,
                )
                offense_rpg = LEAGUE_AVG_RPG
                defense_rpg = LEAGUE_AVG_RPG
            else:
                offense_rpg = (proj.weighted_wrc_plus / 100.0) * LEAGUE_AVG_RPG
                blend_fip = (
                    proj.starter_fip * STARTER_IP + proj.bullpen_fip * BULLPEN_IP
                ) / 9.0
                defense_rpg = (blend_fip / LEAGUE_AVG_FIP) * LEAGUE_AVG_RPG

            park_factor = PARK_FACTORS.get(abbrev, 1.0)

            profiles[team_id] = TeamProfile(
                team_id=team_id,
                team_name=team_name,
                offense_rpg=offense_rpg,
                defense_rpg=defense_rpg,
                park_factor=park_factor,
                rotation=proj.rotation_fips if proj else [],
                bullpen_fip=proj.bullpen_fip if proj else LEAGUE_AVG_FIP,
            )

        return profiles
