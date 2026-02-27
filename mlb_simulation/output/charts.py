"""Matplotlib visualisations for Monte Carlo simulation output."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from mlb_simulation.output.standings import build_standings_df
from mlb_simulation.simulation.season import SeasonProbabilities


def plot_ws_probabilities(
    probabilities: list[SeasonProbabilities],
    output_path: Optional[str | Path] = None,
) -> None:
    """Horizontal bar chart of World Series win probability for all 30 teams.

    Parameters
    ----------
    probabilities:
        Output of ``MonteCarloSeasonSimulator.run()``.
    output_path:
        If given, save the chart to this path instead of showing it
        interactively.
    """
    import matplotlib.pyplot as plt

    df = build_standings_df(probabilities).sort_values("ws_win_pct", ascending=True)

    al_ids = {103}
    colors = [
        "#1f77b4" if row["league_id"] in al_ids else "#d62728"
        for _, row in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(10, 12))
    bars = ax.barh(df["team_name"], df["ws_win_pct"], color=colors)

    ax.set_xlabel("World Series Win Probability (%)")
    ax.set_title("2026 MLB World Series Win Probabilities\n(Monte Carlo Simulation)")
    ax.axvline(100 / 30, color="gray", linestyle="--", linewidth=0.8,
               label="Equal probability (3.3%)")

    # Add value labels on bars
    for bar, val in zip(bars, df["ws_win_pct"]):
        if val >= 0.1:
            ax.text(
                bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8,
            )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="American League"),
        Patch(facecolor="#d62728", label="National League"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_division_race(
    game_results_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    division_id: int,
    output_path: Optional[str | Path] = None,
) -> None:
    """Line chart of cumulative W% over the season for one division.

    Parameters
    ----------
    game_results_df:
        Output of ``SimulationEngine.results_df()`` for a single season.
    teams_df:
        DataFrame from ``MLBClient.get_teams()``.
    division_id:
        Numeric division ID (e.g. 201 for AL East).
    output_path:
        If given, save the chart instead of displaying it.
    """
    import matplotlib.pyplot as plt

    # Identify teams in this division
    div_teams = teams_df[teams_df["division_id"] == division_id]
    div_name = (
        div_teams["division_name"].iloc[0] if not div_teams.empty else str(division_id)
    )
    team_ids = set(div_teams["team_id"].tolist())
    id_to_name = dict(zip(div_teams["team_id"], div_teams["name"]))

    if not team_ids:
        print(f"No teams found for division_id={division_id}")
        return

    # Compute per-team cumulative W% in game order
    wins: dict[int, list[int]] = {t: [] for t in team_ids}
    games_played: dict[int, int] = {t: 0 for t in team_ids}

    for _, row in game_results_df.iterrows():
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        winner_id = int(row["winner_id"])

        for tid in (home_id, away_id):
            if tid in team_ids:
                games_played[tid] += 1
                w = wins[tid][-1] if wins[tid] else 0
                wins[tid].append(w + (1 if tid == winner_id else 0))

    fig, ax = plt.subplots(figsize=(12, 6))

    for tid in team_ids:
        w_list = wins[tid]
        if not w_list:
            continue
        g_list = range(1, len(w_list) + 1)
        wpct = [w / g for w, g in zip(w_list, g_list)]
        ax.plot(g_list, wpct, label=id_to_name.get(tid, str(tid)), linewidth=1.5)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Games Played")
    ax.set_ylabel("Win Percentage")
    ax.set_title(f"2026 {div_name} Division Race (Single Simulation)")
    ax.legend(loc="upper right")
    ax.set_ylim(0.3, 0.7)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)
