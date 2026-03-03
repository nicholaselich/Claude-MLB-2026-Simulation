# Claude MLB 2026 Simulation

A Monte Carlo MLB season simulator built with [Claude Code](https://claude.ai/claude-code). Runs 1,000 full 2026 seasons using FanGraphs Steamer projections to produce win totals, playoff odds, and World Series probabilities for all 30 teams.

## How it works

```
FanGraphs Steamer projections (hitters.json / pitchers.json)
        │
        ▼
ProjectionsAggregator
  PA-weighted wRC+ per team (offense)
  FIP-sorted rotation (GS ≥ 10) stored per team
  Leverage-adjusted bullpen FIP: closer 1.0 IP weight,
    2 setup men 0.5 IP each, middle relief 1.5 IP
        │
        ▼
TeamStrengthModel
  offense_rpg = (wRC+ / 100) × 4.50
  defense_rpg = blend_FIP / 4.20 × 4.50  (per-starter-slot each game)
        │
        ▼
SimulationEngine  ×1000
  Negative-binomial run distribution (r=4)
  Home advantage factor: 1.03×
  Per-game rotation slot tracking (ace → #2 → #3 … across 162 games)
  Extra innings until untied
        │
        ▼
PlayoffSimulator (2022+ bracket)
  Seeds 1–2 bye | Wild Card: 3v6, 4v5
  Division Series (best-of-5) → LCS (best-of-7) → World Series (best-of-7)
  Playoff roster optimization: top-3 starters only, 4.5/4.5 IP split
  Per-game ace reset for each new series
        │
        ▼
Standings table + win ranges (p10–p90) + WS matchup table + charts
```

## 2026 Projections

_1,000-simulation Monte Carlo average — FanGraphs Steamer projections, March 2026_

### American League

**AL East**
| Team | W | L | W% | Div% | Playoff% | WS% | W Range |
|------|---|---|----|------|----------|-----|---------|
| Toronto Blue Jays | 87.9 | 74.1 | .542 | 37.0% | 72.4% | 6.5% | 79–96 |
| New York Yankees | 85.5 | 76.5 | .528 | 23.5% | 62.7% | 5.0% | 77–93 |
| Boston Red Sox | 83.8 | 78.2 | .517 | 16.9% | 51.4% | 4.0% | 75–92 |
| Baltimore Orioles | 83.6 | 78.4 | .516 | 16.9% | 48.6% | 2.5% | 75–92 |
| Tampa Bay Rays | 80.1 | 81.9 | .494 | 5.7% | 28.3% | 0.7% | 72–88 |

**AL Central**
| Team | W | L | W% | Div% | Playoff% | WS% | W Range |
|------|---|---|----|------|----------|-----|---------|
| Detroit Tigers | 84.6 | 77.4 | .522 | 47.7% | 60.1% | 5.2% | 77–92 |
| Kansas City Royals | 80.7 | 81.3 | .498 | 22.3% | 33.9% | 1.1% | 73–88 |
| Minnesota Twins | 79.8 | 82.2 | .493 | 19.7% | 32.3% | 0.7% | 71–88 |
| Cleveland Guardians | 77.5 | 84.5 | .478 | 9.4% | 17.7% | 0.8% | 69–85 |
| Chicago White Sox | 69.4 | 92.6 | .429 | 0.9% | 1.6% | 0.0% | 61–78 |

**AL West**
| Team | W | L | W% | Div% | Playoff% | WS% | W Range |
|------|---|---|----|------|----------|-----|---------|
| Seattle Mariners | 91.6 | 70.4 | .565 | 70.9% | 89.1% | 15.5% | 84–100 |
| Texas Rangers | 83.1 | 78.9 | .513 | 16.4% | 46.9% | 3.1% | 75–91 |
| Houston Astros | 82.0 | 80.0 | .506 | 9.4% | 38.2% | 3.0% | 74–90 |
| Athletics | 75.6 | 86.4 | .467 | 1.8% | 11.2% | 0.4% | 68–84 |
| Los Angeles Angels | 72.4 | 89.6 | .447 | 1.5% | 5.6% | 0.2% | 64–80 |

### National League

**NL East**
| Team | W | L | W% | Div% | Playoff% | WS% | W Range |
|------|---|---|----|------|----------|-----|---------|
| Atlanta Braves | 89.2 | 72.8 | .551 | 45.0% | 77.7% | 9.1% | 81–98 |
| New York Mets | 88.2 | 73.8 | .545 | 35.6% | 73.8% | 5.7% | 80–96 |
| Philadelphia Phillies | 84.1 | 77.9 | .519 | 17.0% | 48.9% | 4.2% | 76–92 |
| Miami Marlins | 76.3 | 85.7 | .471 | 2.1% | 10.2% | 0.0% | 68–85 |
| Washington Nationals | 70.4 | 91.6 | .434 | 0.3% | 1.6% | 0.0% | 63–79 |

**NL Central**
| Team | W | L | W% | Div% | Playoff% | WS% | W Range |
|------|---|---|----|------|----------|-----|---------|
| Chicago Cubs | 83.9 | 78.1 | .518 | 44.4% | 54.6% | 2.3% | 76–92 |
| Milwaukee Brewers | 80.6 | 81.4 | .497 | 22.3% | 32.0% | 0.7% | 73–89 |
| Pittsburgh Pirates | 79.5 | 82.5 | .491 | 18.0% | 27.5% | 1.0% | 71–88 |
| St. Louis Cardinals | 77.0 | 85.0 | .475 | 10.2% | 15.4% | 0.0% | 69–85 |
| Cincinnati Reds | 74.6 | 87.4 | .461 | 5.1% | 8.3% | 0.1% | 67–83 |

**NL West**
| Team | W | L | W% | Div% | Playoff% | WS% | W Range |
|------|---|---|----|------|----------|-----|---------|
| Los Angeles Dodgers | 95.6 | 66.4 | .590 | 74.2% | 96.0% | 20.4% | 87–104 |
| Arizona Diamondbacks | 85.2 | 76.8 | .526 | 10.7% | 54.9% | 2.5% | 78–93 |
| San Francisco Giants | 84.2 | 77.8 | .520 | 7.3% | 51.7% | 2.3% | 76–92 |
| San Diego Padres | 83.9 | 78.1 | .518 | 7.8% | 47.4% | 3.0% | 76–92 |
| Colorado Rockies | 59.7 | 102.3 | .368 | 0.0% | 0.0% | 0.0% | 51–68 |

### World Series odds (all 30 teams)

| Team | WS% |
|------|-----|
| Los Angeles Dodgers | 20.4% |
| Seattle Mariners | 15.5% |
| Atlanta Braves | 9.1% |
| Toronto Blue Jays | 6.5% |
| New York Mets | 5.7% |
| Detroit Tigers | 5.2% |
| New York Yankees | 5.0% |
| Philadelphia Phillies | 4.2% |
| Boston Red Sox | 4.0% |
| Texas Rangers | 3.1% |
| San Diego Padres | 3.0% |
| Houston Astros | 3.0% |
| Baltimore Orioles | 2.5% |
| Arizona Diamondbacks | 2.5% |
| Chicago Cubs | 2.3% |
| San Francisco Giants | 2.3% |
| Kansas City Royals | 1.1% |
| Pittsburgh Pirates | 1.0% |
| Cleveland Guardians | 0.8% |
| Tampa Bay Rays | 0.7% |
| Milwaukee Brewers | 0.7% |
| Minnesota Twins | 0.7% |
| Athletics | 0.4% |
| Los Angeles Angels | 0.2% |
| Cincinnati Reds | 0.1% |
| Chicago White Sox | 0.0% |
| Colorado Rockies | 0.0% |
| Miami Marlins | 0.0% |
| St. Louis Cardinals | 0.0% |
| Washington Nationals | 0.0% |

### Most likely World Series matchups

| AL Champion | NL Champion | Pct |
|-------------|-------------|-----|
| Seattle Mariners | Los Angeles Dodgers | 9.6% |
| Toronto Blue Jays | Los Angeles Dodgers | 5.6% |
| Seattle Mariners | Atlanta Braves | 4.1% |
| Seattle Mariners | New York Mets | 3.1% |
| Detroit Tigers | Los Angeles Dodgers | 3.0% |
| Baltimore Orioles | Los Angeles Dodgers | 2.9% |
| New York Yankees | Los Angeles Dodgers | 2.6% |
| Boston Red Sox | Los Angeles Dodgers | 2.5% |

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
- Divisional standings with win ranges (p10–p90) printed to stdout
- Most likely World Series matchup table (top 8 AL × NL pairs)
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
| `STARTER_IP` | 5.5 | Average starter innings per game (regular season) |
| `BULLPEN_IP` | 3.5 | Average bullpen innings per game (regular season) |
| `PLAYOFF_STARTER_IP` | 4.5 | Starter innings per game in playoffs |
| `PLAYOFF_BULLPEN_IP` | 4.5 | Bullpen innings per game in playoffs |
| `PLAYOFF_ROTATION_DEPTH` | 3 | Number of starters used in playoffs |
