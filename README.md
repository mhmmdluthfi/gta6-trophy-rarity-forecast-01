# GTA VI Trophy Rarity Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mhmmdluthfi/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mhmmdluthfi/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty](https://make.fun/c/gta6-trophy-rarity-forecast-01)
**gta6-trophy-rarity-forecast-01** — forecast, for each of Grand Theft Auto VI's
55 PlayStation 5 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its
Exophase earn rate at 2026-11-26T00:00:00Z, one week after launch. Scored by
mean pinball loss on the logit scale; $250 across the top 20 places.

## How it works

`main.py` (Python 3.13, standard library only, offline, deterministic) reads
every `<id>.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

A trophy's rate is `earners / players_tracked`, and the two move for
different reasons: `players_tracked` grows as new profiles sync (a flood of
new players dilutes every rate), while earners grow only as players actually
reach that trophy. So the median is an ensemble of two fits against
`x = ln(1 + hours since the first read)`:

1. **Counts model** — least-squares fits of `ln(players_tracked)` and
   `ln(earners)` against `x`, extrapolated to the target:
   `rate = 100 * earners / players`. This is what makes slow/grind trophies
   fall while early story trophies hold or rise as the denominator floods.
2. **Logit-trend model** — least-squares fit of `logit(rate)` against `x`,
   read at the target.

Median logit = `0.45 * trend + 0.45 * counts + 0.1 * last read`, with the move
from the last read clamped to ±3.5 logit units.

The spread is the honest part. On launch night Exophase's page sat frozen
at 638 players for hours, then refreshed to 3,868 with every rate but the
100% one lower — a level shift no history shows, and the house baseline's
narrow 90-minute spread scores ~1.51 on the practice for exactly that
reason. So sigma is the quadrature sum of a random-walk term
(`0.5 * sqrt(horizon in days)`), the trend fit's residual scatter times its
prediction-interval factor, a **refresh floor of 0.85 logit**, and a
small-count term (`0.5 / sqrt(earners)` — launch-night rates rest on 4–13
earners). The band therefore grows with horizon and stays wide even when
the history is perfectly flat.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local validation

Practice cases p00/p01 (frozen launch-night history, 638 players):

| check | result |
|---|---|
| output shape (55 rows, 5 quantiles, non-decreasing, 0–100) | valid |
| file size | 2,690 / 2,630 bytes (limit 65,536) |
| runtime | ~0.05 s (limit 8 s) |
| determinism (run twice, diff) | byte-identical |

The kit holds no practice answers (Exophase shows only current rates), so
pinball loss is reported under refresh scenarios built from the one observed
refresh (638 → 3,868 players tracked), mean loss over both practice cases:

| scenario (answer rates) | this entry | house baseline |
|---|---|---|
| earners frozen (rates ÷ 6.06) | **1.13** | 1.69 |
| earners ×2 (rates ÷ 3.03) | **0.60** | 1.01 |
| earners ×4 (rates ÷ 1.52) | **0.26** | 0.33 |
| no refresh (answer = last rate) | 0.18 | 0.02 |

On a synthetic final-like case (361 ten-minute reads over 60 h, players
638 → 38,648, horizon 6,510 min, truth from the same growth law):
loss **0.292** vs baseline 0.323, typical miss 0.023 pp vs 0.076 pp.

No network, no model calls, no secrets in this repository.
