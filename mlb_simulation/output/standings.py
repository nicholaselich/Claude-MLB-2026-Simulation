"""Print and build standings tables from Monte Carlo probabilities."""

from __future__ import annotations

import pandas as pd

from mlb_simulation.simulation.season import SeasonProbabilities


def build_standings_df(probabilities: list[SeasonProbabilities]) -> pd.DataFrame:
    """Convert a list of ``SeasonProbabilities`` to a machine-readable DataFrame."""
    rows = []
    for sp in probabilities:
        rows.append(
            {
                "team_id": sp.team_id,
                "team_name": sp.team_name,
                "league_id": sp.league_id,
                "league_name": sp.league_name,
                "division_id": sp.division_id,
                "division_name": sp.division_name,
                "avg_wins": round(sp.avg_wins, 1),
                "avg_losses": round(sp.avg_losses, 1),
                "win_pct": round(sp.avg_wins / (sp.avg_wins + sp.avg_losses), 3)
                if (sp.avg_wins + sp.avg_losses) > 0
                else 0.0,
                "p10_wins": int(round(sp.p10_wins)),
                "p90_wins": int(round(sp.p90_wins)),
                "div_title_pct": round(sp.div_title_pct, 1),
                "playoff_pct": round(sp.playoff_pct, 1),
                "ws_win_pct": round(sp.ws_win_pct, 1),
            }
        )
    return pd.DataFrame(rows)


def print_standings(
    probabilities: list[SeasonProbabilities],
    teams_df: pd.DataFrame | None = None,
) -> None:
    """Print a division-by-division standings table to stdout.

    Example output::

        AL East              W      L    W%   Div%  Playoff%   WS%
        New York Yankees   94.3   67.7  .582  62.3%    88.1%  14.2%
    """
    df = build_standings_df(probabilities)
    if df.empty:
        print("No probabilities to display.")
        return

    # Group by division
    league_order = ["American League", "National League"]
    divisions_by_league: dict[str, list[str]] = {}
    for _, row in df.drop_duplicates("division_name").iterrows():
        lg = str(row["league_name"])
        divisions_by_league.setdefault(lg, []).append(str(row["division_name"]))

    for league in league_order:
        divisions = sorted(divisions_by_league.get(league, []))
        for division in divisions:
            div_df = (
                df[df["division_name"] == division]
                .sort_values("avg_wins", ascending=False)
                .reset_index(drop=True)
            )
            _print_division_table(division, div_df)
    print()


def _print_division_table(division: str, df: pd.DataFrame) -> None:
    header = f"\n{'─' * 72}"
    col_header = (
        f"{'Team':<26} {'W':>5} {'L':>5}  {'W%':>5}  "
        f"{'Div%':>6}  {'Playoff%':>9}  {'WS%':>5}  {'Range':>7}"
    )
    print(header)
    print(f" {division}")
    print(f"{'─' * 72}")
    print(col_header)
    print(f"{'─' * 72}")
    for _, row in df.iterrows():
        w_pct = f".{int(row['win_pct'] * 1000):03d}"
        win_range = f"{row['p10_wins']}–{row['p90_wins']}"
        print(
            f" {row['team_name']:<25} "
            f"{row['avg_wins']:>5.1f} "
            f"{row['avg_losses']:>5.1f}  "
            f"{w_pct:>5}  "
            f"{row['div_title_pct']:>5.1f}%  "
            f"{row['playoff_pct']:>8.1f}%  "
            f"{row['ws_win_pct']:>4.1f}%  "
            f"{win_range:>7}"
        )
