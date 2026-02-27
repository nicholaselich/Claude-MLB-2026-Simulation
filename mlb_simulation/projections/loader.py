"""Load FanGraphs Steamer projection JSON files from the data/ directory."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# FanGraphs abbreviation → MLB Stats API abbreviation
_ABBREV_MAP: dict[str, str] = {
    "ARI": "AZ",
    "CHW": "CWS",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "WSN": "WSH",
}


class ProjectionsLoader:
    """Load FanGraphs Steamer projections from JSON files.

    Parameters
    ----------
    data_dir:
        Directory containing ``hitters.json`` and ``pitchers.json``.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._data_dir = Path(data_dir)

    def _normalise_team(self, abbrev) -> str | None:
        if not isinstance(abbrev, str) or not abbrev.strip():
            return None
        return _ABBREV_MAP.get(abbrev.strip(), abbrev.strip())

    def _load_json(self, filename: str) -> list[dict]:
        path = self._data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Projection file not found: {path}. "
                "Place hitters.json and pitchers.json in the data/ directory."
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_batting(self) -> pd.DataFrame:
        """Load batting projections, normalise team abbreviations.

        Returns
        -------
        DataFrame with columns: PlayerName, Team, League, PA, wRC+, playerid
        """
        records = self._load_json("hitters.json")
        df = pd.DataFrame(records)
        df["Team"] = df["Team"].apply(self._normalise_team)
        # Drop free agents and players with no team assignment
        df = df[df["Team"].notna()].copy()
        df["PA"] = pd.to_numeric(df["PA"], errors="coerce").fillna(0)
        df["wRC+"] = pd.to_numeric(df["wRC+"], errors="coerce").fillna(100)
        required = ["PlayerName", "Team", "League", "PA", "wRC+", "playerid"]
        return df[required].reset_index(drop=True)

    def load_pitching(self) -> pd.DataFrame:
        """Load pitching projections, normalise team abbreviations.

        Returns
        -------
        DataFrame with columns: PlayerName, Team, League, IP, GS, ERA, FIP, playerid
        """
        records = self._load_json("pitchers.json")
        df = pd.DataFrame(records)
        df["Team"] = df["Team"].apply(self._normalise_team)
        df = df[df["Team"].notna()].copy()
        df["IP"] = pd.to_numeric(df["IP"], errors="coerce").fillna(0)
        df["GS"] = pd.to_numeric(df["GS"], errors="coerce").fillna(0)
        df["ERA"] = pd.to_numeric(df["ERA"], errors="coerce").fillna(4.20)
        df["FIP"] = pd.to_numeric(df["FIP"], errors="coerce").fillna(4.20)
        required = ["PlayerName", "Team", "League", "IP", "GS", "ERA", "FIP", "playerid"]
        return df[required].reset_index(drop=True)
