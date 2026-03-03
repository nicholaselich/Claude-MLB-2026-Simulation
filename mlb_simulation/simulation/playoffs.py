"""Playoff bracket simulator using the 2022+ MLB format.

Format (per league):
  Wild Card Series (best-of-3): seeds 3 vs 6, 4 vs 5  — seeds 1 & 2 have byes
  Division Series  (best-of-5): 1 vs lower seed WC winner, 2 vs higher seed WC winner
  League Championship (best-of-7)
  World Series    (best-of-7): AL champ vs NL champ
"""

from __future__ import annotations

import numpy as np

from mlb_simulation.data.models import TeamProfile
from mlb_simulation.strength.team_model import (
    HOME_ADV_FACTOR, LEAGUE_AVG_RPG, game_defense_rpg,
    PLAYOFF_STARTER_IP, PLAYOFF_BULLPEN_IP, PLAYOFF_ROTATION_DEPTH,
)

# Home/away pattern per game slot for each series format.
# True = higher seed (team_a) is at home; False = lower seed (team_b) is at home.
_HOME_PATTERN: dict[int, list[bool]] = {
    3: [True, True, False],                             # Wild Card (best-of-3)
    5: [True, True, False, False, True],                # Division Series (best-of-5)
    7: [True, True, False, False, False, True, True],   # LCS / World Series (best-of-7)
}


class PlayoffSimulator:
    """Simulate an MLB playoff bracket to determine the World Series winner.

    Parameters
    ----------
    team_profiles:
        Dict mapping ``team_id`` → ``TeamProfile``.
    """

    def __init__(self, team_profiles: dict[int, TeamProfile]) -> None:
        self._profiles = team_profiles

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def simulate_bracket(
        self, al_seeds: list[int], nl_seeds: list[int]
    ) -> dict:
        """Simulate the full playoff bracket.

        Parameters
        ----------
        al_seeds / nl_seeds:
            Six team IDs ordered by seed (index 0 = #1 seed / best record).
            Seeds 1-2 receive first-round byes; seeds 3-6 play Wild Card.

        Returns
        -------
        Dict with bracket results including ``world_series_winner``.
        """
        bracket: dict = {"AL": {}, "NL": {}}

        for league, seeds in (("AL", al_seeds), ("NL", nl_seeds)):
            # seeds[0] = #1, seeds[1] = #2, ..., seeds[5] = #6
            # Wild Card (best-of-3): 3 vs 6, 4 vs 5  (seeds[2] vs seeds[5], seeds[3] vs seeds[4])
            wc_a_winner = self._simulate_series(seeds[2], seeds[5], 3)  # 3 vs 6
            wc_b_winner = self._simulate_series(seeds[3], seeds[4], 3)  # 4 vs 5
            bracket[league]["wild_card"] = [wc_a_winner, wc_b_winner]

            # Re-seed the 4 DS participants by original seed rank
            ds_field = sorted(
                [seeds[0], seeds[1], wc_a_winner, wc_b_winner],
                key=lambda t: seeds.index(t),
            )
            # Division Series (best-of-5): #1 vs lowest seed, #2 vs remaining
            ds_a_winner = self._simulate_series(ds_field[0], ds_field[3], 5)
            ds_b_winner = self._simulate_series(ds_field[1], ds_field[2], 5)
            bracket[league]["division_series"] = [ds_a_winner, ds_b_winner]

            # League Championship Series (best-of-7): re-seed 2 survivors
            lcs_field = sorted(
                [ds_a_winner, ds_b_winner],
                key=lambda t: seeds.index(t),
            )
            lcs_winner = self._simulate_series(lcs_field[0], lcs_field[1], 7)
            bracket[league]["lcs_winner"] = lcs_winner

        # World Series (best-of-7): AL champ hosts first (seeds[0] advantage heuristic)
        al_champ = bracket["AL"]["lcs_winner"]
        nl_champ = bracket["NL"]["lcs_winner"]
        ws_winner = self._simulate_series(al_champ, nl_champ, 7)
        bracket["world_series_winner"] = ws_winner

        return bracket

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _game_win_prob(
        self, team_a_id: int, team_b_id: int, team_a_home: bool, game_num: int = 0
    ) -> float:
        """Win probability for team_a in a single game.

        Uses rotation-slot-aware defense: game_num=0 → ace, game_num=1 → #2, etc.
        Both teams reset to their ace (slot 0) for game 1 of each series.
        """
        prof_a = self._profiles.get(team_a_id)
        prof_b = self._profiles.get(team_b_id)

        if prof_a and prof_b:
            def_rpg_a = game_defense_rpg(
                prof_a, game_num, PLAYOFF_STARTER_IP, PLAYOFF_BULLPEN_IP, PLAYOFF_ROTATION_DEPTH
            )
            def_rpg_b = game_defense_rpg(
                prof_b, game_num, PLAYOFF_STARTER_IP, PLAYOFF_BULLPEN_IP, PLAYOFF_ROTATION_DEPTH
            )
            exp_a = prof_a.offense_rpg * (def_rpg_b / LEAGUE_AVG_RPG)
            exp_b = prof_b.offense_rpg * (def_rpg_a / LEAGUE_AVG_RPG)
            if team_a_home:
                exp_a *= HOME_ADV_FACTOR
            else:
                exp_b *= HOME_ADV_FACTOR
            exponent = 1.83
            p_a = exp_a ** exponent / (exp_a ** exponent + exp_b ** exponent)
        else:
            p_a = 0.55 if team_a_home else 0.45

        return float(np.clip(p_a, 0.01, 0.99))

    def _simulate_series(
        self, team_a_id: int, team_b_id: int, games_in_series: int
    ) -> int:
        """Simulate a best-of-N series; return the winning team ID.

        ``team_a`` is assumed to be the higher seed (home in games 1-2).
        """
        wins_needed = (games_in_series // 2) + 1
        home_pattern = _HOME_PATTERN.get(games_in_series, [True] * games_in_series)

        wins_a = wins_b = 0
        game_num = 0

        while wins_a < wins_needed and wins_b < wins_needed:
            team_a_home = home_pattern[game_num % len(home_pattern)]
            p_a = self._game_win_prob(team_a_id, team_b_id, team_a_home, game_num)
            if np.random.random() < p_a:
                wins_a += 1
            else:
                wins_b += 1
            game_num += 1

        return team_a_id if wins_a >= wins_needed else team_b_id
