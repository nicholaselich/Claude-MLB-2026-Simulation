"""Plotly figure builders for the web dashboard.

Pure functions — no Flask dependency. Accept pre-computed data from the pipeline.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go

_AL_COLOUR = "#3b82f6"
_NL_COLOUR = "#ef4444"
_AL_ID = 103


def _team_colour(league_id: int) -> str:
    return _AL_COLOUR if league_id == _AL_ID else _NL_COLOUR


def build_ws_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of WS win probability per team."""
    df_sorted = df.sort_values("ws_win_pct", ascending=True)
    colours = [_team_colour(int(lid)) for lid in df_sorted["league_id"]]

    fig = go.Figure(
        go.Bar(
            orientation="h",
            x=df_sorted["ws_win_pct"],
            y=df_sorted["team_name"],
            marker_color=colours,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(
        x=100 / 30,
        line_dash="dash",
        opacity=0.4,
        annotation_text="Expected",
        annotation_position="top",
    )
    fig.update_layout(
        template="plotly_dark",
        height=700,
        xaxis_title="WS Win Probability (%)",
        yaxis_title="",
        margin=dict(l=180, r=20, t=20, b=40),
        showlegend=False,
    )
    return fig


def build_win_range_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal range-bar chart showing P10–P90 win projections per team."""
    df_sorted = df.sort_values("avg_wins", ascending=True)
    colours = [_team_colour(int(lid)) for lid in df_sorted["league_id"]]

    # Invisible base bar to create the left offset
    invisible = go.Bar(
        orientation="h",
        x=df_sorted["p10_wins"],
        y=df_sorted["team_name"],
        marker_color="rgba(0,0,0,0)",
        hoverinfo="skip",
        showlegend=False,
    )

    # Visible range bar (width = p90 - p10, base = p10)
    range_bar = go.Bar(
        orientation="h",
        x=df_sorted["p90_wins"] - df_sorted["p10_wins"],
        y=df_sorted["team_name"],
        base=df_sorted["p10_wins"],
        marker_color=colours,
        opacity=0.6,
        name="P10–P90",
        hovertemplate=(
            "%{y}<br>Range: %{base:.0f}–%{x:.0f} wins<extra></extra>"
        ),
    )

    # Average wins dot
    avg_dot = go.Scatter(
        x=df_sorted["avg_wins"],
        y=df_sorted["team_name"],
        mode="markers",
        marker=dict(color="white", size=8),
        name="Avg",
        hovertemplate="%{y}: %{x:.1f} avg wins<extra></extra>",
    )

    fig = go.Figure(data=[invisible, range_bar, avg_dot])
    fig.update_layout(
        template="plotly_dark",
        height=700,
        barmode="overlay",
        xaxis=dict(range=[55, 115], title="Projected Wins"),
        yaxis_title="",
        margin=dict(l=180, r=20, t=20, b=40),
    )
    return fig


def build_matchup_heatmap(
    ws_matchup_counts: dict,
    team_info: dict,
    n_simulations: int,
    top_n: int = 8,
) -> go.Figure:
    """Heatmap of top-N AL vs NL WS matchup probabilities."""
    al_totals: dict[int, int] = defaultdict(int)
    nl_totals: dict[int, int] = defaultdict(int)
    for (al_id, nl_id), count in ws_matchup_counts.items():
        al_totals[al_id] += count
        nl_totals[nl_id] += count

    top_al = sorted(al_totals, key=lambda k: al_totals[k], reverse=True)[:top_n]
    top_nl = sorted(nl_totals, key=lambda k: nl_totals[k], reverse=True)[:top_n]

    def _name(team_id: int) -> str:
        return team_info.get(team_id, {}).get("name", f"Team {team_id}")

    al_names = [_name(t) for t in top_al]
    nl_names = [_name(t) for t in top_nl]

    matrix = [
        [
            ws_matchup_counts.get((al_id, nl_id), 0) / n_simulations * 100
            for nl_id in top_nl
        ]
        for al_id in top_al
    ]
    text = [[f"{v:.1f}%" for v in row] for row in matrix]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=nl_names,
            y=al_names,
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hovertemplate="AL: %{y}<br>NL: %{x}<br>Prob: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=550,
        yaxis=dict(autorange="reversed"),
        xaxis_title="NL Champion",
        yaxis_title="AL Champion",
        margin=dict(l=160, r=20, t=20, b=120),
    )
    return fig


def build_division_tables_html(df: pd.DataFrame) -> tuple[str, str]:
    """Return (al_html, nl_html) — Bootstrap table markup per league."""

    def _division_block(div_name: str, div_df: pd.DataFrame) -> str:
        rows = ""
        for _, row in div_df.sort_values("avg_wins", ascending=False).iterrows():
            win_range = f"{row['p10_wins']}–{row['p90_wins']}"
            w_pct = f".{int(round(float(row['win_pct']) * 1000)):03d}"
            rows += (
                f"<tr>"
                f"<td>{row['team_name']}</td>"
                f"<td>{row['avg_wins']:.1f}</td>"
                f"<td>{row['avg_losses']:.1f}</td>"
                f"<td>{w_pct}</td>"
                f"<td>{row['div_title_pct']:.1f}%</td>"
                f"<td>{row['playoff_pct']:.1f}%</td>"
                f"<td>{row['ws_win_pct']:.1f}%</td>"
                f"<td>{win_range}</td>"
                f"</tr>"
            )
        header = (
            "<thead><tr>"
            "<th>Team</th><th>W</th><th>L</th><th>W%</th>"
            "<th>Div%</th><th>Playoff%</th><th>WS%</th><th>Range</th>"
            "</tr></thead>"
        )
        return (
            f'<h5 class="text-light mt-3">{div_name}</h5>'
            f'<table class="table table-dark table-sm table-hover">'
            f"{header}<tbody>{rows}</tbody></table>"
        )

    al_html = ""
    nl_html = ""

    al_df = df[df["league_id"] == _AL_ID]
    nl_df = df[df["league_id"] != _AL_ID]

    for div_name in sorted(al_df["division_name"].unique()):
        al_html += _division_block(div_name, al_df[al_df["division_name"] == div_name])

    for div_name in sorted(nl_df["division_name"].unique()):
        nl_html += _division_block(div_name, nl_df[nl_df["division_name"] == div_name])

    return al_html, nl_html
