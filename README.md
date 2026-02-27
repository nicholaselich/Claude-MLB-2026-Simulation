# Claude MLB 2026 Simulation

A Monte Carlo MLB season simulator built with [Claude Code](https://claude.ai/claude-code). Runs 1,000 full 2026 seasons using FanGraphs Steamer projections to produce win totals, playoff odds, and World Series probabilities for all 30 teams.

## How it works

```
FanGraphs Steamer projections (hitters.json / pitchers.json)
        │
        ▼
ProjectionsAggregator
  PA-weighted wRC+ per team (offense)
  IP-weighted FIP per team  (starter + bullpen)
        │
        ▼
TeamStrengthModel
  offense_rpg = (wRC+ / 100) × 4.50
  defense_rpg = blend_FIP / 4.20 × 4.50
        │
        ▼
SimulationEngine  ×1000
  Negative-binomial run distribution (r=4)
  Home advantage factor: 1.03×
  Extra innings until untied
        │
        ▼
PlayoffSimulator (2022+ bracket)
  Seeds 1–2 bye | Wild Card: 3v6, 4v5
  Division Series (best-of-5) → LCS (best-of-7) → World Series (best-of-7)
  Pythagorean series win probabilities
        │
        ▼
Standings table + charts
```

## Sample output

```
────────────────────────────────────────────────────────────
 National League West
────────────────────────────────────────────────────────────
Team                           W     L     W%    Div%   Playoff%    WS%
────────────────────────────────────────────────────────────
 Los Angeles Dodgers        93.4  68.6   .577  69.2%     93.4%  12.2%
 Arizona Diamondbacks       84.6  77.4   .522  12.6%     51.8%   3.9%
 San Francisco Giants       83.3  78.7   .514  10.0%     48.5%   3.2%
 San Diego Padres           82.9  79.1   .512   8.2%     43.1%   3.0%
 Colorado Rockies           61.0 101.0   .377   0.0%      0.0%   0.0%
```

## Setup

**Requirements:** Python 3.10+

```bash
git clone git@github.com:nicholaselich/Claude-MLB-2026-Simulation.git
cd Claude-MLB-2026-Simulation
pip install -r requirements.txt
```

**Projection files** (not included in repo — download from FanGraphs):

Place the following in the `data/` directory:
- `data/hitters.json` — Steamer batting projections
- `data/pitchers.json` — Steamer pitching projections

Export as JSON from the FanGraphs projections leaderboard. The loader expects these fields:

| File | Required fields |
|------|----------------|
| `hitters.json` | `PlayerName`, `Team`, `League`, `PA`, `wRC+`, `playerid` |
| `pitchers.json` | `PlayerName`, `Team`, `League`, `IP`, `GS`, `FIP`, `ERA`, `playerid` |

## Usage

```bash
python3 main.py
```

Outputs:
- Divisional standings printed to stdout
- `simulation_results.csv` — full 30-team probability table
- `ws_probabilities.png` — World Series odds bar chart (AL blue / NL red)
- `al_east_race.png` — AL East division race line chart (single season)

## Project structure

```
mlb_simulation/
  api/            MLBClient — fetches 2026 schedule and team metadata
  data/           Schedule, TeamStats, GameResult, TeamProfile dataclasses
  projections/    ProjectionsLoader + ProjectionsAggregator
  strength/       TeamStrengthModel (projections → runs-per-game profiles)
  simulation/     SimulationEngine, PlayoffSimulator, MonteCarloSeasonSimulator
  output/         Standings tables and matplotlib charts
main.py           Full pipeline entry point
```

## Key constants

Defined in `mlb_simulation/strength/team_model.py` — override to adjust the model:

| Constant | Default | Description |
|----------|---------|-------------|
| `LEAGUE_AVG_RPG` | 4.50 | League-average runs per game |
| `LEAGUE_AVG_FIP` | 4.20 | League-average FIP |
| `HOME_ADV_FACTOR` | 1.03 | Multiplicative home run-scoring boost |
| `STARTER_IP` | 5.5 | Average starter innings per game |
| `BULLPEN_IP` | 3.5 | Average bullpen innings per game |
