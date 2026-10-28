# AI Appendix: Phase 4

*Model and date:* 10/27 - Claude Opus 5.5

*Version 1 prompt:*

[Prep] Act as an experimentation platform engineer who has implemented
sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true
effects. Implement an always-valid sequential test -- a mixture
sequential probability ratio test or an alpha-spending boundary -- that
allows continuous monitoring without inflating the Type I error rate.
Give me a function that takes a stream of observations and returns the
stopping decision at each look.

[Iterate] numpy and scipy only. Reuse my log structure. The function
must be testable against a pure-null stream.

[Mechanism Check] Explain why the naive approach of testing at every
look inflates Type I error, and precisely what the always-valid
construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams and report the empirical
false positive rate. If it is not close to 5%, tell me what is wrong.

*Reply:*

"""Always-valid two-sample mSPRT (normal mixture, Johari et al. 2017). numpy + scipy only."""
import numpy as np
from scipy import stats

def msprt_stream(treat, ctrl, alpha=0.05, tau2=None, sigma2=None, burn_in=50):
    """Paired stream (x_i, y_i). Returns one log record per look.
    sigma2: known per-arm variance; if None, plug-in running estimate after burn_in.
    tau2: mixture variance on effect; default = sigma2 (a reasonable scale choice)."""
    t, c = np.asarray(treat, float), np.asarray(ctrl, float)
    n = np.arange(1, len(t) + 1)
    d = t - c
    dbar = np.cumsum(d) / n
    if sigma2 is None:  # running pooled variance estimate
        m2 = lambda x: (np.cumsum(x**2) - n * (np.cumsum(x) / n)**2) / np.maximum(n - 1, 1)
        s2 = 0.5 * (m2(t) + m2(c))
    else:
        s2 = np.full(len(t), float(sigma2))
    s2 = np.maximum(s2, 1e-12)          # guard n=1 zero variance
    V = 2 * s2                                   # variance of one paired difference
    tau2 = s2 if tau2 is None else np.full(len(t), float(tau2))
    log_lam = 0.5 * np.log(V / (V + n * tau2)) + (n**2 * tau2 * dbar**2) / (2 * V * (V + n * tau2))
    log_lam[n <= burn_in] = -np.inf               # no decisions during burn-in
    p_av = np.minimum.accumulate(np.minimum(1.0, np.exp(-log_lam)))  # always-valid p
    half_w = np.sqrt(V * (V + n * tau2) / (n**2 * tau2) * (np.log((V + n * tau2) / V) - 2 * np.log(alpha)))
    stopped = False
    log = []
    for i in range(len(t)):
        reject = (not stopped) and p_av[i] <= alpha
        stopped = stopped or reject
        log.append({"look": int(n[i]), "effect_est": float(dbar[i]), "p_always_valid": float(p_av[i]),
                    "ci_low": float(dbar[i] - half_w[i]), "ci_high": float(dbar[i] + half_w[i]),
                    "decision": "reject" if reject else ("stopped" if stopped else "continue")})
    return log

def first_rejection(log):
    return next((r["look"] for r in log if r["decision"] == "reject"), None)

def naive_reject_any(treat, ctrl, alpha=0.05, burn_in=50):
    """Fixed-horizon z-test applied at every look (the broken approach)."""
    t, c = np.asarray(treat, float), np.asarray(ctrl, float)
    n = np.arange(1, len(t) + 1)
    s2 = 0.5 * (np.var(t[:burn_in], ddof=1) + np.var(c[:burn_in], ddof=1))
    z = (np.cumsum(t - c) / n) / np.sqrt(2 * s2 / n)
    return np.any(np.abs(z[burn_in:]) > stats.norm.ppf(1 - alpha / 2))

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    R, N = 1000, 5000
    fp_known = fp_plug = fp_naive = 0
    for _ in range(R):
        x, y = rng.normal(0, 1, N), rng.normal(0, 1, N)      # pure null
        fp_known += first_rejection(msprt_stream(x, y, sigma2=1.0)) is not None
        fp_plug  += first_rejection(msprt_stream(x, y)) is not None
        fp_naive += naive_reject_any(x, y)
    se = lambda k: np.sqrt(k / R * (1 - k / R) / R)
    for name, k in [("mSPRT known sigma", fp_known), ("mSPRT plug-in sigma", fp_plug), ("naive z every look", fp_naive)]:
        print(f"{name:22s} FPR = {k/R:.4f}  (SE {se(k):.4f})")

## What version 1 left unsaid or got wrong

Details about the log

## Version 2: the prompt as sent
```
[Prep] Act as an experimentation platform engineer who has implemented
sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true
effects. Implement an always-valid sequential test -- a mixture
sequential probability ratio test or an alpha-spending boundary -- that
allows continuous monitoring without inflating the Type I error rate.
Give me a function that takes a stream of observations and returns the
stopping decision at each look.

[Iterate] numpy and scipy only. Reuse my log structure. The function
must be testable against a pure-null stream.

What my data is: my tests measure revenue per order in dollars with an SD of 20; the effect is the treatment minus control.
Test size - each test is between 1000 and 100 000 customers, and my log has a column 'n_tested' with that size
Client checks- Client checks results after every 1000 customers.
How -> run 1000 test streams with my test sizes. Give each a different rng seed and get back and report false positive rates with its standard error


[Mechanism Check] Explain why the naive approach of testing at every
look inflates Type I error, and precisely what the always-valid
construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams and report the empirical
false positive rate. If it is not close to 5%, tell me what is wrong.
```

## Version 2: the reply

"""Always-valid sequential A/B test (Gaussian-mixture SPRT, Johari et al. 2017).
numpy + scipy only. Metric: revenue per order, known SD per arm = 20.
Effect = treatment mean - control mean.
"""
import numpy as np
from scipy.stats import norm

SIGMA = 20.0
LOOK_EVERY = 1000
ALPHA = 0.05


def msprt_stream(values, arms, sigma=SIGMA, tau=0.5, alpha=ALPHA,
                 look_every=LOOK_EVERY):
    """Stream of customers -> decision at each look.

    values : revenue per customer (dollars)
    arms   : 1 = treatment, 0 = control
    tau    : SD of N(0, tau^2) mixing prior on the effect (dollars); set it
             to the typical |true effect| in your portfolio.
    Returns a structured array, one row per look.
    """
    values, arms = np.asarray(values, float), np.asarray(arms, int)
    looks = np.arange(look_every, len(values) + 1, look_every)
    if len(values) % look_every:
        looks = np.append(looks, len(values))

    cs_t = np.cumsum(values * arms)[looks - 1]
    cs_c = np.cumsum(values * (1 - arms))[looks - 1]
    n_t = np.cumsum(arms)[looks - 1]
    n_c = looks - n_t
    ok = (n_t > 0) & (n_c > 0)

    diff = np.where(ok, cs_t / np.maximum(n_t, 1) - cs_c / np.maximum(n_c, 1), 0.0)
    V = np.where(ok, sigma**2 * (1 / np.maximum(n_t, 1) + 1 / np.maximum(n_c, 1)), np.inf)

    # Mixture likelihood ratio: integral of N(diff; theta, V) / N(diff; 0, V) over N(0, tau^2)
    log_lam = np.where(ok, 0.5 * np.log(V / (V + tau**2))
                       + tau**2 * diff**2 / (2 * V * (V + tau**2)), 0.0)
    p_av = np.minimum.accumulate(np.minimum(1.0, np.exp(-log_lam)))  # always-valid p
    reject = p_av <= alpha
    stop = np.cumsum(reject) > 0                                     # absorbing

    z = np.where(ok, diff / np.sqrt(V), 0.0)
    naive = np.cumsum(2 * norm.sf(np.abs(z)) <= alpha) > 0           # for comparison

    out = np.zeros(len(looks), dtype=[("look", int), ("n_seen", int), ("diff", float),
                                      ("se", float), ("log_lambda", float), ("p_always_valid", float),
                                      ("stop", bool), ("naive_stop", bool)])
    out["look"], out["n_seen"], out["diff"], out["se"] = np.arange(1, len(looks) + 1), looks, diff, np.sqrt(V)
    out["log_lambda"], out["p_always_valid"], out["stop"], out["naive_stop"] = log_lam, p_av, stop, naive
    return out


def simulate_stream(n_tested, effect, seed, sigma=SIGMA, base=50.0):
    rng = np.random.default_rng(seed)
    arms = rng.integers(0, 2, n_tested)
    values = base + effect * arms + rng.normal(0, sigma, n_tested)
    return values, arms


def null_fpr(n_tested_log, n_streams=1000, tau=0.5, seed0=0):
    """Run n_streams pure-null streams, sizes resampled from your log's n_tested."""
    rng = np.random.default_rng(seed0)
    sizes = rng.choice(np.asarray(n_tested_log), n_streams, replace=True)
    rej = np.zeros(n_streams, bool)
    naive = np.zeros(n_streams, bool)
    for i, n in enumerate(sizes):
        res = msprt_stream(*simulate_stream(int(n), 0.0, seed=seed0 + 1 + i), tau=tau)
        rej[i], naive[i] = res["stop"][-1], res["naive_stop"][-1]
    se = lambda p: np.sqrt(p * (1 - p) / n_streams)
    return {"msprt_fpr": rej.mean(), "msprt_se": se(rej.mean()),
            "naive_fpr": naive.mean(), "naive_se": se(naive.mean())}


if __name__ == "__main__":
    # Placeholder for your log's n_tested column (200 tests, 1k-100k).
    # Replace with: n_tested_log = log["n_tested"]
    n_tested_log = np.random.default_rng(42).integers(1, 101, 200) * 1000
    r = null_fpr(n_tested_log, n_streams=1000)
    print(f"mSPRT  FPR = {r['msprt_fpr']:.4f}  (SE {r['msprt_se']:.4f})")
    print(f"Naive  FPR = {r['naive_fpr']:.4f}  (SE {r['naive_se']:.4f})")

## Changes I made to the model's code
