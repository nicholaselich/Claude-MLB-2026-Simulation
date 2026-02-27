"""Client for the MLB Stats API (statsapi.mlb.com)."""

import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://statsapi.mlb.com/api/v1"
_SPORT_ID = 1  # MLB


class MLBClient:
    """Thin wrapper around the public MLB Stats REST API.

    All methods return pandas DataFrames so downstream code can stay
    in a single data-manipulation idiom.  Raw JSON is cached on the
    instance so repeated calls don't hit the network twice.
    """

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{_BASE_URL}/{path.lstrip('/')}"
        logger.debug("GET %s  params=%s", url, params)
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def get_schedule(
        self,
        season: int = 2026,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return the regular-season schedule as a DataFrame.

        Parameters
        ----------
        season:
            Four-digit MLB season year.
        start_date / end_date:
            Optional ISO-8601 date strings (``"YYYY-MM-DD"``) to narrow
            the window.  When omitted the full season is returned.

        Returns
        -------
        DataFrame with columns:
            game_id, game_date, game_time, status,
            home_team_id, home_team_name,
            away_team_id, away_team_name,
            venue_id, venue_name, series_description
        """
        params: dict = {
            "sportId": _SPORT_ID,
            "season": season,
            "gameType": "R",  # Regular season only
        }
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        data = self._get("schedule", params)

        rows = []
        for date_block in data.get("dates", []):
            for game in date_block.get("games", []):
                rows.append(
                    {
                        "game_id": game["gamePk"],
                        "game_date": game["gameDate"][:10],
                        "game_time": game["gameDate"][11:16],
                        "status": game["status"]["detailedState"],
                        "home_team_id": game["teams"]["home"]["team"]["id"],
                        "home_team_name": game["teams"]["home"]["team"]["name"],
                        "away_team_id": game["teams"]["away"]["team"]["id"],
                        "away_team_name": game["teams"]["away"]["team"]["name"],
                        "venue_id": game.get("venue", {}).get("id"),
                        "venue_name": game.get("venue", {}).get("name"),
                        "series_description": game.get("seriesDescription"),
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["game_date"] = pd.to_datetime(df["game_date"])
            df.sort_values("game_date", inplace=True)
            df.reset_index(drop=True, inplace=True)
        logger.info("Loaded %d games for %d season.", len(df), season)
        return df

    # ------------------------------------------------------------------
    # Teams
    # ------------------------------------------------------------------

    def get_teams(self) -> pd.DataFrame:
        """Return all active MLB teams.

        Returns
        -------
        DataFrame with columns:
            team_id, name, abbreviation, league_id, league_name,
            division_id, division_name, venue_id, venue_name
        """
        data = self._get("teams", {"sportId": _SPORT_ID, "activeStatus": "Y"})

        rows = []
        for t in data.get("teams", []):
            rows.append(
                {
                    "team_id": t["id"],
                    "name": t["name"],
                    "abbreviation": t.get("abbreviation"),
                    "league_id": t.get("league", {}).get("id"),
                    "league_name": t.get("league", {}).get("name"),
                    "division_id": t.get("division", {}).get("id"),
                    "division_name": t.get("division", {}).get("name"),
                    "venue_id": t.get("venue", {}).get("id"),
                    "venue_name": t.get("venue", {}).get("name"),
                }
            )

        df = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
        logger.info("Loaded %d teams.", len(df))
        return df

    # ------------------------------------------------------------------
    # Team stats
    # ------------------------------------------------------------------

    def _get_team_stats(
        self, team_id: int, season: int, group: str
    ) -> pd.DataFrame:
        """Fetch hitting or pitching season stats for a single team."""
        data = self._get(
            f"teams/{team_id}/stats",
            {
                "stats": "season",
                "season": season,
                "group": group,
                "sportId": _SPORT_ID,
            },
        )

        splits = []
        for stat_block in data.get("stats", []):
            for split in stat_block.get("splits", []):
                row = {"team_id": team_id, "season": season, "group": group}
                row.update(split.get("stat", {}))
                splits.append(row)

        return pd.DataFrame(splits)

    def get_team_hitting_stats(
        self, team_id: int, season: int = 2026
    ) -> pd.DataFrame:
        """Season-aggregate hitting stats for *team_id*."""
        return self._get_team_stats(team_id, season, "hitting")

    def get_team_pitching_stats(
        self, team_id: int, season: int = 2026
    ) -> pd.DataFrame:
        """Season-aggregate pitching stats for *team_id*."""
        return self._get_team_stats(team_id, season, "pitching")

    def get_all_team_stats(self, season: int = 2026) -> dict[str, pd.DataFrame]:
        """Fetch hitting and pitching stats for every MLB team.

        Returns a dict with keys ``"hitting"`` and ``"pitching"``, each
        holding a DataFrame with one row per team.
        """
        teams = self.get_teams()
        hitting_frames = []
        pitching_frames = []

        for _, row in teams.iterrows():
            tid = int(row["team_id"])
            try:
                h = self._get_team_stats(tid, season, "hitting")
                if not h.empty:
                    h["team_name"] = row["name"]
                    hitting_frames.append(h)

                p = self._get_team_stats(tid, season, "pitching")
                if not p.empty:
                    p["team_name"] = row["name"]
                    pitching_frames.append(p)

            except requests.HTTPError as exc:
                logger.warning("Could not load stats for team %d: %s", tid, exc)

        hitting = (
            pd.concat(hitting_frames, ignore_index=True)
            if hitting_frames
            else pd.DataFrame()
        )
        pitching = (
            pd.concat(pitching_frames, ignore_index=True)
            if pitching_frames
            else pd.DataFrame()
        )

        logger.info(
            "Loaded stats for %d hitting / %d pitching rows.",
            len(hitting),
            len(pitching),
        )
        return {"hitting": hitting, "pitching": pitching}
