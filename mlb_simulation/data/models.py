"""Lightweight wrappers that add domain-specific helpers on top of raw DataFrames.

These classes do *not* own the network layer — they accept DataFrames that
were produced by ``MLBClient`` and provide convenient access patterns so
the simulation engine doesn't need to know about raw column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


class Schedule:
    """Wraps the schedule DataFrame with game-lookup helpers.

    Parameters
    ----------
    df:
        DataFrame produced by ``MLBClient.get_schedule()``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def games_on(self, date: str) -> pd.DataFrame:
        """Return all games scheduled on *date* (``"YYYY-MM-DD"``)."""
        target = pd.Timestamp(date)
        return self._df[self._df["game_date"] == target].reset_index(drop=True)

    def games_for_team(self, team_id: int) -> pd.DataFrame:
        """Return every game involving *team_id* (home or away)."""
        mask = (self._df["home_team_id"] == team_id) | (
            self._df["away_team_id"] == team_id
        )
        return self._df[mask].reset_index(drop=True)

    def unplayed_games(self) -> pd.DataFrame:
        """Return games whose status indicates they haven't been played yet."""
        not_final = ~self._df["status"].str.lower().str.contains("final|completed")
        return self._df[not_final].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"<Schedule games={len(self)}>"


class TeamStats:
    """Wraps hitting and pitching DataFrames with per-team lookup helpers.

    Parameters
    ----------
    hitting:
        DataFrame produced by ``MLBClient.get_all_team_stats()["hitting"]``.
    pitching:
        DataFrame produced by ``MLBClient.get_all_team_stats()["pitching"]``.
    """

    def __init__(self, hitting: pd.DataFrame, pitching: pd.DataFrame) -> None:
        self._hitting = hitting.copy()
        self._pitching = pitching.copy()

    @property
    def hitting(self) -> pd.DataFrame:
        return self._hitting

    @property
    def pitching(self) -> pd.DataFrame:
        return self._pitching

    def hitting_for(self, team_id: int) -> pd.Series:
        """Return the hitting stat row for *team_id*, or an empty Series."""
        rows = self._hitting[self._hitting["team_id"] == team_id]
        return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)

    def pitching_for(self, team_id: int) -> pd.Series:
        """Return the pitching stat row for *team_id*, or an empty Series."""
        rows = self._pitching[self._pitching["team_id"] == team_id]
        return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)

    def stat(self, team_id: int, group: str, column: str, default=None):
        """Generic stat accessor.

        Parameters
        ----------
        team_id:
            MLB team ID.
        group:
            ``"hitting"`` or ``"pitching"``.
        column:
            Column name in the relevant DataFrame (e.g. ``"era"``, ``"ops"``).
        default:
            Value to return when the stat is missing.
        """
        row = self.hitting_for(team_id) if group == "hitting" else self.pitching_for(team_id)
        return row.get(column, default)

    def __repr__(self) -> str:
        return (
            f"<TeamStats hitting_rows={len(self._hitting)} "
            f"pitching_rows={len(self._pitching)}>"
        )


@dataclass
class GameResult:
    """Stores the outcome of a single simulated game.

    The simulation engine will produce instances of this class; the
    season aggregator will consume them.
    """

    game_id: int
    home_team_id: int
    away_team_id: int
    home_score: int
    away_score: int
    innings: int = 9
    metadata: dict = field(default_factory=dict)

    @property
    def winner_id(self) -> int:
        return self.home_team_id if self.home_score > self.away_score else self.away_team_id

    @property
    def loser_id(self) -> int:
        return self.away_team_id if self.home_score > self.away_score else self.home_team_id

    @property
    def is_home_win(self) -> bool:
        return self.home_score > self.away_score

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "game_id": self.game_id,
                "home_team_id": self.home_team_id,
                "away_team_id": self.away_team_id,
                "home_score": self.home_score,
                "away_score": self.away_score,
                "innings": self.innings,
                "winner_id": self.winner_id,
            }
        )


@dataclass
class TeamProfile:
    """Runs-per-game offensive and defensive strength for a single team."""

    team_id: int
    team_name: str
    offense_rpg: float    # projected runs scored per game (park-neutral)
    defense_rpg: float    # projected runs allowed per game (park-neutral)
    park_factor: float = 1.0  # venue run-scoring multiplier; applied to both teams

    def pythag_wpct(self, exponent: float = 1.83) -> float:
        o, d = self.offense_rpg ** exponent, self.defense_rpg ** exponent
        return o / (o + d) if (o + d) > 0 else 0.5
