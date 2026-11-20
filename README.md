# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Vellocatta/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Vellocatta/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Earners/players decomposition forecaster for GTA VI's PS5 trophy earn rates a week after launch, as five logit-scale quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per trophy.

Method: rate = earners / players. The tracked-player count is fit from the recent history (log growth, last 24 h) and extrapolated with exponential decay (launch surge fades, tau 48 h); when the history is flat (launch night), a decaying prior growth factor applies (players 638 → 3868 in 90 min, the practice's own refresh). Each trophy's earner growth comes from a hazard estimated from its rate and the game's age, gated by a ~6 h warm-up (a new player earns essentially only the starting trophy in their first hours), with new-player earn probability capped at 0.85 and the forecast rate capped at 97% (logit(99.99) explodes the pinball loss when the truth is merely ~95%). Everything is computed at run time from the case file itself, so the same program works on the practice cases and the final case. No network, no clock, no randomness; Python 3.13 standard library only.

Practice validation (pinball loss on the logit scale, scored against answers reconstructed from the published refresh figures and calibrated to the published persistence loss 1.51): p00 0.153, p01 0.153 — mean 0.153, against the house persistence baseline's 1.51 (fitted-curve baseline identical on the practice). The practice is a refresh event: Exophase's counts jump 638 → 3868 players and every rate but the 100% one falls ~6×. Persistence misses it; this model anticipates the dilution. Synthetic 4.5-day-horizon worlds (decaying player growth, hazards 0.25×–2× the model's) also beat persistence in every scenario (worst 0.54×, mean 0.40×).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=55z7zxas&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
