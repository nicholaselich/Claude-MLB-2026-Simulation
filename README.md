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

## 2026 Projections

_1,000-simulation Monte Carlo average — FanGraphs Steamer projections, March 2026_

### American League

**AL East**
| Team | W | L | W% | Div% | Playoff% | WS% |
|------|---|---|----|------|----------|-----|
| Toronto Blue Jays | 87.9 | 74.1 | .543 | 35.2% | 73.5% | 7.8% |
| New York Yankees | 85.5 | 76.5 | .528 | 22.4% | 61.0% | 5.1% |
| Baltimore Orioles | 84.4 | 77.6 | .521 | 18.6% | 54.4% | 3.9% |
| Boston Red Sox | 83.8 | 78.2 | .517 | 18.1% | 52.6% | 3.7% |
| Tampa Bay Rays | 79.1 | 82.9 | .488 | 5.7% | 22.1% | 0.7% |

**AL Central**
| Team | W | L | W% | Div% | Playoff% | WS% |
|------|---|---|----|------|----------|-----|
| Detroit Tigers | 84.2 | 77.8 | .520 | 46.0% | 59.2% | 4.4% |
| Kansas City Royals | 81.1 | 80.9 | .501 | 23.7% | 36.4% | 2.0% |
| Minnesota Twins | 80.6 | 81.4 | .498 | 21.0% | 33.1% | 1.4% |
| Cleveland Guardians | 76.8 | 85.2 | .474 | 8.1% | 14.8% | 0.5% |
| Chicago White Sox | 69.8 | 92.2 | .431 | 1.2% | 2.8% | 0.0% |

**AL West**
| Team | W | L | W% | Div% | Playoff% | WS% |
|------|---|---|----|------|----------|-----|
| Seattle Mariners | 90.9 | 71.1 | .561 | 70.3% | 89.0% | 13.4% |
| Texas Rangers | 82.8 | 79.2 | .511 | 13.0% | 43.2% | 2.9% |
| Houston Astros | 82.4 | 79.6 | .509 | 13.2% | 42.3% | 2.3% |
| Athletics | 76.4 | 85.6 | .472 | 2.7% | 12.1% | 0.1% |
| Los Angeles Angels | 72.2 | 89.8 | .446 | 0.8% | 3.5% | 0.0% |

### National League

**NL East**
| Team | W | L | W% | Div% | Playoff% | WS% |
|------|---|---|----|------|----------|-----|
| Atlanta Braves | 90.0 | 72.0 | .555 | 48.7% | 82.1% | 9.1% |
| New York Mets | 88.6 | 73.4 | .547 | 37.6% | 76.0% | 7.4% |
| Philadelphia Phillies | 83.3 | 78.7 | .514 | 12.5% | 45.0% | 3.6% |
| Miami Marlins | 76.3 | 85.7 | .471 | 1.0% | 9.8% | 0.3% |
| Washington Nationals | 70.6 | 91.4 | .436 | 0.2% | 1.6% | 0.0% |

**NL Central**
| Team | W | L | W% | Div% | Playoff% | WS% |
|------|---|---|----|------|----------|-----|
| Chicago Cubs | 84.4 | 77.6 | .521 | 45.9% | 57.5% | 2.7% |
| Milwaukee Brewers | 80.5 | 81.5 | .497 | 21.9% | 33.3% | 1.0% |
| Pittsburgh Pirates | 79.8 | 82.2 | .493 | 17.5% | 28.3% | 0.3% |
| St. Louis Cardinals | 77.9 | 84.1 | .481 | 10.8% | 20.0% | 0.7% |
| Cincinnati Reds | 74.8 | 87.2 | .462 | 3.9% | 7.9% | 0.0% |

**NL West**
| Team | W | L | W% | Div% | Playoff% | WS% |
|------|---|---|----|------|----------|-----|
| Los Angeles Dodgers | 94.9 | 67.1 | .586 | 76.0% | 95.8% | 18.4% |
| Arizona Diamondbacks | 84.7 | 77.3 | .523 | 10.6% | 54.8% | 3.3% |
| San Francisco Giants | 83.8 | 78.2 | .517 | 8.1% | 48.5% | 3.1% |
| San Diego Padres | 82.2 | 79.8 | .507 | 5.3% | 39.4% | 1.9% |
| Colorado Rockies | 60.1 | 101.9 | .371 | 0.0% | 0.0% | 0.0% |

### World Series odds (all 30 teams)

| Team | WS% |
|------|-----|
| Los Angeles Dodgers | 18.4% |
| Seattle Mariners | 13.4% |
| Atlanta Braves | 9.1% |
| Toronto Blue Jays | 7.8% |
| New York Mets | 7.4% |
| New York Yankees | 5.1% |
| Detroit Tigers | 4.4% |
| Baltimore Orioles | 3.9% |
| Boston Red Sox | 3.7% |
| Philadelphia Phillies | 3.6% |
| Arizona Diamondbacks | 3.3% |
| San Francisco Giants | 3.1% |
| Texas Rangers | 2.9% |
| Chicago Cubs | 2.7% |
| Houston Astros | 2.3% |
| Kansas City Royals | 2.0% |
| San Diego Padres | 1.9% |
| Minnesota Twins | 1.4% |
| Milwaukee Brewers | 1.0% |
| St. Louis Cardinals | 0.7% |
| Tampa Bay Rays | 0.7% |
| Cleveland Guardians | 0.5% |
| Miami Marlins | 0.3% |
| Pittsburgh Pirates | 0.3% |
| Athletics | 0.1% |
| Chicago White Sox | 0.0% |
| Cincinnati Reds | 0.0% |
| Colorado Rockies | 0.0% |
| Los Angeles Angels | 0.0% |
| Washington Nationals | 0.0% |

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
