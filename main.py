"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase earn
rate at the case target as five quantiles (5/25/50/75/95%), scored by pinball
loss on the logit scale.

Model (written in-session, standard library only, deterministic, no network):

The rate of a trophy is earners / players_tracked, and the two move for
different reasons: players_tracked grows as new profiles sync (a flood of
new players dilutes every rate), while earners grow only as players actually
reach that trophy.  Forecasting the rate directly confuses the two, so the
median is an ensemble of two views of the same history, both fitted against
x = ln(1 + hours since the first read) -- a concave clock, so growth
decelerates the way launch-week growth does:

  1. counts model: least-squares fits of ln(players_tracked) and ln(earners)
     against x, extrapolated to the target, rate = 100 * earners / players.
     This is the mechanism that makes slow/grind trophies fall while early
     story trophies hold or rise as the denominator floods.
  2. logit-trend model: least-squares fit of logit(rate) against x (the
     house curve's shape, fitted here over all reads), read at the target.

median logit = 0.45 * trend + 0.45 * counts + 0.1 * last read (persistence
anchor, so a wild extrapolation is damped), with the move from the last read
clamped to +/-3.5 logit units.

Spread (the practice lesson: the house baseline scores ~1.51 because its
90-minute spread is far too narrow across an Exophase refresh -- the page
sat frozen at 638 players for hours, then refreshed to 3,868 with every
rate but the 100% one lower; a refresh is a level shift the history never
shows).  sigma is the quadrature sum of:

  - a random-walk term 0.5 * sqrt(horizon in days) (the house baseline's);
  - the trend fit's residual scatter, times the standard prediction-interval
    factor for extrapolating to the target (floored, so a frozen history
    still carries uncertainty);
  - a refresh floor of 0.65 logit: Exophase refreshes lag and jump, and no
    history, frozen or not, can see the next one coming;
  - a small-count term 0.5 / sqrt(earners in the last read): launch-night
    rates rest on 4-13 earners, so their logit is noisy by construction.

The spread therefore grows with horizon (the final's 4.52 days get a much
wider band than a 90-minute practice case) and is wide even when the
history is perfectly flat.  Quantiles are the logistic of
median logit + z_tau * sigma, rounded to 4 decimals half-up.

Contract: read every <id>.in in $MAKEFUN_INPUT_DIR (else /input), write
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output): {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]}, one row per trophy, each finite
0..100, non-decreasing, file <= 65,536 bytes.  Runs offline in well under
8 s; two runs are byte-identical (pure arithmetic, no randomness).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Ensemble weights for the median logit: trend / counts / persistence.
# The persistence anchor is light (0.1): in a moving regime it drags the
# median toward stale rates, and the +/-3.5 logit move clamp already guards
# wild extrapolations.
W_TREND, W_COUNTS, W_LAST = 0.45, 0.45, 0.1
# Widest logit move the median may make away from the last read.
MAX_LOGIT_MOVE = 3.5
# Random-walk spread, logit units per sqrt(day) (the house baseline's).
SIGMA_RW = 0.5
# Refresh floor: logit spread for the Exophase refresh jump no history shows.
# A 0.85 floor says a refresh can move a rate ~2.4x either way at the 5%
# quantile; the one refresh the house observed moved the denominator 6x.
SIGMA_REFRESH = 0.85
# Fit-residual spread is floored and capped; prediction factor is capped.
FIT_SIGMA_MIN, FIT_SIGMA_MAX = 0.15, 1.2
PRED_FACTOR_MAX = 2.5
SIGMA_MAX = 2.5
# Denominator growth guard: players tracked may not be forecast to grow by
# more than this factor over the last read (launch week is fast, not magic).
PLAYERS_GROWTH_CAP = 40.0


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def logistic(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up (as JavaScript Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to ~1e-9; used so any levels list works, not just the scored
    five.  Deterministic and stdlib-only.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(text):
    """Parse an ISO-8601 timestamp to an aware datetime, or None."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def least_squares(xs, ys):
    """Least-squares line y ~ a + b*x.  Returns (a, b, resid_std, n, xbar, sxx)
    or None when the fit is degenerate (fewer than 2 points or no x spread).
    """
    n = len(xs)
    if n < 2:
        return None
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return (a, b, resid, n, xbar, sxx)


def forecast_case(case: dict, name: str) -> dict:
    levels = list(case.get("levels") or LEVELS)
    zetas = [norm_ppf(p) for p in levels]
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0

    # Time axis: hours since the first read, on a log clock.  Reads before
    # release (launch night starts 23:50Z, release is midnight) sit at x=0.
    t0 = parse_time(history[0]["at"]) if history else None
    target = parse_time(case.get("target"))
    if t0 is None:
        t0 = parse_time(case.get("at"))
    if t0 is None:
        t0 = datetime(2026, 11, 19, tzinfo=timezone.utc)
    if target is None:
        horizon_min = case.get("horizon_minutes") or 0
        target = t0 + timedelta(minutes=horizon_min)

    def x_of(when):
        dt = parse_time(when)
        if dt is None:
            return 0.0
        hours = max(0.0, (dt - t0).total_seconds() / 3600.0)
        return math.log1p(hours)

    x_target = x_of(case.get("target"))
    xs, ps = [], []
    for read in history:
        p = read.get("players_tracked")
        if isinstance(p, (int, float)) and p > 0:
            xs.append(x_of(read.get("at")))
            ps.append(math.log(p))
    p_fit = least_squares(xs, ps)
    p_last = ps[-1] if ps else math.log(638.0)
    if p_fit is not None:
        log_p_target = p_fit[0] + p_fit[1] * x_target
        # Never forecast the denominator to shrink, nor to explode.
        log_p_target = min(math.log(PLAYERS_GROWTH_CAP) + p_last,
                           max(p_last, log_p_target))
    else:
        log_p_target = p_last
    players_target = math.exp(log_p_target)

    rows = []
    for i, trophy in enumerate(trophies):
        # Series for this trophy: (x, rate, earners) over reads that have it.
        tx, tlogit, elogx, elogy = [], [], [], []
        last_rate, last_earners = None, None
        for read in history:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            r = rates[i]
            p = read.get("players_tracked")
            if not isinstance(r, (int, float)) or not isinstance(p, (int, float)) or p <= 0:
                continue
            last_rate = r
            last_earners = r / 100.0 * p
            x = x_of(read.get("at"))
            if 0 < r < 100:
                tx.append(x)
                tlogit.append(logit(r))
            # Earners on a log scale; 0 earners read as half an earner so a
            # trophy nobody has yet still has a (very low) fitted level.
            elogx.append(x)
            elogy.append(math.log(max(0.5, r / 100.0 * p)))
        if last_rate is None:
            last_rate = 1.0
            last_earners = 6.0
        last_logit = logit(last_rate)

        # 1. Logit trend: read the fitted line at the target; flat when the
        # history cannot support a slope (frozen launch-night reads).
        trend_fit = least_squares(tx, tlogit)
        if trend_fit is not None:
            trend_logit = trend_fit[0] + trend_fit[1] * x_target
            a, b, resid, n, xbar, sxx = trend_fit
            pred_factor = math.sqrt(1.0 + 1.0 / n + (x_target - xbar) ** 2 / sxx)
        else:
            trend_logit = last_logit
            resid, pred_factor = 0.0, 1.0
        fit_sigma = min(FIT_SIGMA_MAX, max(FIT_SIGMA_MIN, resid)) * min(pred_factor, PRED_FACTOR_MAX)

        # 2. Counts model: extrapolate earners and the denominator apart.
        e_fit = least_squares(elogx, elogy)
        if e_fit is not None:
            log_e_target = e_fit[0] + e_fit[1] * x_target
        else:
            log_e_target = math.log(max(0.5, last_earners or 0.5))
        earners_target = min(players_target, math.exp(log_e_target))
        counts_rate = clamp(100.0 * earners_target / players_target)
        counts_logit = logit(counts_rate)

        # Median: ensemble, damped toward persistence, move clamped.
        median_logit = (W_TREND * trend_logit + W_COUNTS * counts_logit
                        + W_LAST * last_logit)
        median_logit = min(last_logit + MAX_LOGIT_MOVE,
                           max(last_logit - MAX_LOGIT_MOVE, median_logit))

        # Spread: random walk + fit scatter + refresh floor + small counts.
        rw = SIGMA_RW * math.sqrt(max(0.0, horizon_days))
        count_term = 0.5 / math.sqrt(max(1.0, last_earners or 1.0))
        sigma = math.sqrt(rw * rw + fit_sigma * fit_sigma
                          + SIGMA_REFRESH * SIGMA_REFRESH
                          + count_term * count_term)
        sigma = min(SIGMA_MAX, sigma)

        qs = []
        prev = None
        for z in zetas:
            q = round4(clamp(logistic(median_logit + z * sigma)))
            # Rounding can only tie, never invert; enforce it explicitly.
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
