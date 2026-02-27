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

            profiles[team_id] = TeamProfile(
                team_id=team_id,
                team_name=team_name,
                offense_rpg=offense_rpg,
                defense_rpg=defense_rpg,
            )

        return profiles
