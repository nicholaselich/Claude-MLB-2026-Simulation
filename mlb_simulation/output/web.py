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

_TEAM_COLOURS: dict[str, str] = {
    # AL East
    "Baltimore Orioles":    "#DF4601",
    "Boston Red Sox":       "#BD3039",
    "New York Yankees":     "#1D428A",
    "Tampa Bay Rays":       "#8FBCE6",
    "Toronto Blue Jays":    "#134A8E",
    # AL Central
    "Chicago White Sox":    "#C4CED4",
    "Cleveland Guardians":  "#E31937",
    "Detroit Tigers":       "#FA4616",
    "Kansas City Royals":   "#004687",
    "Minnesota Twins":      "#D31145",
    # AL West
    "Houston Astros":       "#EB6E1F",
    "Los Angeles Angels":   "#BA0021",
    "Oakland Athletics":    "#003831",
    "Las Vegas Athletics":  "#003831",
    "Seattle Mariners":     "#005C5C",
    "Texas Rangers":        "#003278",
    # NL East
    "Atlanta Braves":       "#CE1141",
    "Miami Marlins":        "#00A3E0",
    "New York Mets":        "#002D72",
    "Philadelphia Phillies": "#E81828",
    "Washington Nationals": "#AB0003",
    # NL Central
    "Chicago Cubs":         "#0E3386",
    "Cincinnati Reds":      "#C6011F",
    "Milwaukee Brewers":    "#FFC52F",
    "Pittsburgh Pirates":   "#FDB827",
    "St. Louis Cardinals":  "#C41E3A",
    # NL West
    "Arizona Diamondbacks": "#A71930",
    "Colorado Rockies":     "#33006F",
    "Los Angeles Dodgers":  "#005A9C",
    "San Diego Padres":     "#FFC425",
    "San Francisco Giants": "#FD5A1E",
}


def _team_colour(team_name: str = "", league_id: int = 0) -> str:
    if team_name in _TEAM_COLOURS:
        return _TEAM_COLOURS[team_name]
    return _AL_COLOUR if league_id == _AL_ID else _NL_COLOUR


def build_ws_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of WS win probability per team."""
    df_sorted = df.sort_values("ws_win_pct", ascending=True)
    colours = [_team_colour(row["team_name"], int(row["league_id"])) for _, row in df_sorted.iterrows()]

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
    colours = [_team_colour(row["team_name"], int(row["league_id"])) for _, row in df_sorted.iterrows()]

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
            team = str(row['team_name']).replace("'", "\\'")
            rows += (
                f"<tr style='cursor:pointer' onclick=\"showTeamModal('{team}')\">"
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


def build_scatter_chart(profiles_data: list[dict]) -> go.Figure:
    """Offense RPG vs Defense RPG scatter; bubble size = WS%."""
    names = [d["team_name"] for d in profiles_data]
    off = [d["offense_rpg"] for d in profiles_data]
    defn = [d["defense_rpg"] for d in profiles_data]
    ws = [d["ws_win_pct"] for d in profiles_data]
    colours = [_team_colour(d["team_name"], d["league_id"]) for d in profiles_data]
    sizes = [max(8, w * 4) for w in ws]

    fig = go.Figure(
        go.Scatter(
            x=off,
            y=defn,
            mode="markers+text",
            text=names,
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(color=colours, size=sizes, opacity=0.8, line=dict(width=1, color="rgba(255,255,255,0.25)")),
            customdata=list(zip(names, ws)),
            hovertemplate="<b>%{customdata[0]}</b><br>Offense: %{x:.2f} RPG<br>Defense: %{y:.2f} RPG<br>WS%: %{customdata[1]:.1f}%<extra></extra>",
        )
    )
    # Quadrant lines at league average
    avg = 4.50
    fig.add_hline(y=avg, line_dash="dash", opacity=0.3)
    fig.add_vline(x=avg, line_dash="dash", opacity=0.3)
    # Quadrant labels
    for txt, ax, ay in [
        ("Elite Offense<br>Elite Pitching", 5.2, 3.9),
        ("Elite Offense<br>Weak Pitching", 5.2, 5.1),
        ("Weak Offense<br>Elite Pitching", 3.8, 3.9),
        ("Weak Offense<br>Weak Pitching", 3.8, 5.1),
    ]:
        fig.add_annotation(text=txt, x=ax, y=ay, showarrow=False,
                           font=dict(size=9, color="#444"), align="center")
    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_title="Offense (RPG)",
        yaxis_title="Defense (RPG allowed — lower is better)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=60, r=20, t=20, b=60),
        showlegend=False,
    )
    return fig


def build_division_race_json(results_df, teams_df: pd.DataFrame) -> str:
    """Return JSON string: {division_name: [{name, color, x, y}, ...]} for client-side rendering."""
    import json as _json

    results = results_df.sort_values("game_date") if "game_date" in results_df.columns else results_df

    team_meta = {
        int(r["team_id"]): (r["name"], r["division_name"], int(r["league_id"]))
        for _, r in teams_df.iterrows()
    }

    out: dict = {}
    for division in sorted(teams_df["division_name"].dropna().unique()):
        traces = []
        for tid, (name, div, league_id) in team_meta.items():
            if div != division:
                continue
            colour = _team_colour(name, league_id)
            mask = (results["home_team_id"] == tid) | (results["away_team_id"] == tid)
            team_games = results[mask].copy()
            if team_games.empty:
                continue
            team_games = team_games.sort_values("game_date") if "game_date" in team_games.columns else team_games
            team_games["won"] = (team_games["winner_id"] == tid).astype(int)
            team_games["game_num"] = range(1, len(team_games) + 1)
            team_games["cum_wins"] = team_games["won"].cumsum()
            team_games["win_pct"] = team_games["cum_wins"] / team_games["game_num"]
            traces.append({
                "name": name,
                "color": colour,
                "x": team_games["game_num"].tolist(),
                "y": [round(v, 4) for v in team_games["win_pct"].tolist()],
            })
        out[division] = traces
    return _json.dumps(out)


def build_division_race_charts(results_df, teams_df: pd.DataFrame) -> dict:
    """Return {division_name: go.Figure} — cumulative win-% line chart per division."""
    results = results_df.sort_values("game_date")

    # Build team_id → (name, division_name, league_id) lookup
    team_meta = {
        int(r["team_id"]): (r["name"], r["division_name"], int(r["league_id"]))
        for _, r in teams_df.iterrows()
    }

    charts = {}
    divisions = teams_df["division_name"].dropna().unique()

    for division in sorted(divisions):
        div_team_ids = [
            tid for tid, (name, div, lid) in team_meta.items()
            if div == division
        ]
        traces = []
        for tid in div_team_ids:
            name, _, league_id = team_meta[tid]
            colour = _team_colour(name, league_id)
            team_games = results[
                (results["home_team_id"] == tid) | (results["away_team_id"] == tid)
            ].copy()
            team_games["won"] = (team_games["winner_id"] == tid).astype(int)
            team_games = team_games.sort_values("game_date")
            team_games["game_num"] = range(1, len(team_games) + 1)
            team_games["cum_wins"] = team_games["won"].cumsum()
            team_games["win_pct"] = team_games["cum_wins"] / team_games["game_num"]
            traces.append(go.Scatter(
                x=team_games["game_num"],
                y=team_games["win_pct"],
                mode="lines",
                name=name,
                line=dict(color=colour, width=2),
                hovertemplate=f"<b>{name}</b><br>Game %{{x}}<br>Win%%: %{{y:.3f}}<extra></extra>",
            ))
        fig = go.Figure(traces)
        fig.add_hline(y=0.500, line_dash="dash", opacity=0.3)
        fig.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Game Number",
            yaxis_title="Win %",
            yaxis=dict(tickformat=".3f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=20, t=40, b=60),
        )
        charts[division] = fig

    return charts


def build_playoff_bracket_html(
    df: pd.DataFrame,
    wc_wins: dict,
    ds_wins: dict,
    n_simulations: int,
    team_info: dict,
) -> str:
    """Return an HTML string showing playoff round probabilities per team."""
    _AL_ID_LOCAL = 103

    def _pct(team_id: int, counts: dict) -> float:
        return counts.get(team_id, 0) / n_simulations * 100

    def _league_rows(league_id: int) -> str:
        league_df = df[df["league_id"] == league_id].sort_values("playoff_pct", ascending=False)
        rows = ""
        for _, row in league_df.iterrows():
            tid = int(row["team_id"])
            playoff = row["playoff_pct"]
            wc = _pct(tid, wc_wins)
            ds = _pct(tid, ds_wins)
            ws = row["ws_win_pct"]
            bar_style = "background:#3b82f6;height:6px;border-radius:3px"
            if league_id != _AL_ID_LOCAL:
                bar_style = bar_style.replace("#3b82f6", "#ef4444")

            def bar(val, max_val=100):
                w = max(1, int(val / max_val * 100))
                return f'<div style="{bar_style};width:{w}%"></div>'

            rows += f"""
            <tr onclick="showTeamModal('{row['team_name']}')" style="cursor:pointer">
              <td style="font-weight:500">{row['team_name']}</td>
              <td><div>{playoff:.0f}%</div>{bar(playoff)}</td>
              <td><div>{wc:.0f}%</div>{bar(wc)}</td>
              <td><div>{ds:.0f}%</div>{bar(ds)}</td>
              <td><div>{ws:.1f}%</div>{bar(ws, 20)}</td>
            </tr>"""
        return rows

    al_rows = _league_rows(103)
    nl_rows = _league_rows(104)

    return f"""
    <div class="row g-4">
      <div class="col-lg-6">
        <h6 style="color:#3b82f6">American League</h6>
        <table class="table table-dark table-sm table-hover">
          <thead><tr>
            <th>Team</th><th>Playoff%</th><th>WC Win%</th><th>DS Win%</th><th>WS%</th>
          </tr></thead>
          <tbody>{al_rows}</tbody>
        </table>
      </div>
      <div class="col-lg-6">
        <h6 style="color:#ef4444">National League</h6>
        <table class="table table-dark table-sm table-hover">
          <thead><tr>
            <th>Team</th><th>Playoff%</th><th>WC Win%</th><th>DS Win%</th><th>WS%</th>
          </tr></thead>
          <tbody>{nl_rows}</tbody>
        </table>
      </div>
    </div>"""
