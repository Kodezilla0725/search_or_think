"""Rewards: plain GRPO, fixed per search penalty, SAAS, and Soft-SAAS (Eq. 1 of the proposal).

One definition per quantity, used in training, relabelling and analysis alike:
  p(q)   share of the K search disabled rollouts whose answer covers a gold answer (scoring.knows, cover match)
  N_min  fewest searches among search enabled rollouts with F1 = 1 on the answer inside <answer> tags (is_correct)
  gate   a search penalty applies only to rollouts with F1 = 1 on the answer inside <answer> tags (is_correct)
SAAS and Soft-SAAS read the same p(q) and N_min, so the hard vs soft boundary is the only difference between them.
The released SAAS code uses EM for its labels; we keep cover match for p(q) and store an EM based p(q) next to it.
"""
from dataclasses import dataclass
from typing import Optional

MAX_SEARCHES = 4
ALPHA = 0.05  # SAAS: alpha in the paper (App. B.1), beta = 0.05 in search/utils.py of the released code

# NoSearch needs delta of K search disabled rollouts correct. SAAS uses delta = 2 of 4 in the paper (Sec. 3.1) and in
# the released code (head_correct_count >= 2 with disable_search_first_n = 4); as a ratio it carries over to K = 8.
NOSEARCH_MIN_P = 0.5
# The paper and the released code disagree on NeedSearch. Paper: no search disabled rollout is correct and some
# search enabled one is. Code (generate_with_search.py, mq_group_reward_func): fewer than delta are correct and some
# search enabled one is. We follow the code by default; "paper" stays available for the comparison.
SAAS_RULE = "code"


def is_correct(f1):
    return f1 >= 1.0 - 1e-9


@dataclass
class Boundary:
    p: float              # p(q)
    n_min: Optional[int]  # N_min (None if no search enabled rollout was correct)
    label: str            # SAAS label: NoSearch / NeedSearch / Undetermined


def saas_label(p, n_min, rule=SAAS_RULE, nosearch_min_p=NOSEARCH_MIN_P):
    if rule not in ("code", "paper"):
        raise ValueError(f"unknown SAAS rule {rule!r}")
    if p >= nosearch_min_p:
        return "NoSearch"
    if n_min is not None and (rule == "code" or p == 0):
        return "NeedSearch"
    return "Undetermined"


def estimate_boundary(disabled_correct, enabled_rollouts, rule=SAAS_RULE, nosearch_min_p=NOSEARCH_MIN_P):
    """disabled_correct: list[bool] (cover match); enabled_rollouts: list of (correct: bool, n_search: int) (F1 = 1)."""
    p = sum(disabled_correct) / len(disabled_correct) if disabled_correct else 0.0
    ns = [n for c, n in enabled_rollouts if c]
    n_min = min(ns) if ns else None
    return Boundary(p, n_min, saas_label(p, n_min, rule, nosearch_min_p))


def penalty_saas(n_search, b):
    if b.label == "NoSearch":
        return float(n_search)
    if b.label == "NeedSearch":
        return float(max(0, n_search - b.n_min))
    return 0.0  # Undetermined questions get no search signal


def penalty_soft(n_search, b, max_searches=MAX_SEARCHES):
    n_min = max_searches if b.n_min is None else b.n_min
    return b.p * n_search + (1 - b.p) * max(0, n_search - n_min)


def reward_plain(f1, n_search):
    return f1


def reward_fixed(f1, n_search, lam):
    return f1 - lam * n_search


def reward_saas(f1, n_search, b, alpha, w=1.0):
    return f1 - (alpha * w * penalty_saas(n_search, b) if is_correct(f1) else 0.0)


def reward_soft_saas(f1, n_search, b, alpha, w=1.0, max_searches=MAX_SEARCHES):
    return f1 - (alpha * w * penalty_soft(n_search, b, max_searches) if is_correct(f1) else 0.0)


def compute_rewards(kind, f1s, n_searches, boundaries, *, lam=0.1, alpha=ALPHA, w=1.0, format_bonus=0.0,
                    format_ok=None):
    """Rewards for a batch of rollouts. Returns (rewards, penalised), penalised[i] = rollout i lost reward for searching.
    format_bonus: reward for a well formed but wrong answer (F1 = 0), as in the released SAAS code (0.1 there)."""
    rewards, penalised = [], []
    for i, (f, n) in enumerate(zip(f1s, n_searches)):
        if kind == "plain":
            pen = 0.0
        elif kind == "fixed":
            pen = lam * w * n
        elif kind == "saas":
            pen = alpha * w * penalty_saas(n, boundaries[i]) if is_correct(f) else 0.0
        elif kind == "soft":
            pen = alpha * w * penalty_soft(n, boundaries[i]) if is_correct(f) else 0.0
        else:
            raise ValueError(f"unknown reward {kind!r}")
        bonus = format_bonus if format_bonus and f == 0 and format_ok is not None and format_ok[i] else 0.0
        rewards.append(f + bonus - pen)
        penalised.append(pen > 0)
    return rewards, penalised


def tighten_n_min(boundaries, f1s, n_searches, G, rule=SAAS_RULE):
    """On-policy N_min, as in the released SAAS code: within each group of G rollouts of one question, N_min is at
    most the fewest searches of the group's own correct rollouts. p(q) is left as it was measured."""
    out = list(boundaries)
    for g in range(0, len(out), G):
        ns = [n for f, n in zip(f1s[g:g + G], n_searches[g:g + G]) if is_correct(f)]
        if not ns:
            continue
        for i in range(g, min(g + G, len(out))):
            b = out[i]
            n_min = min(ns) if b.n_min is None else min(b.n_min, min(ns))
            out[i] = Boundary(b.p, n_min, saas_label(b.p, n_min, rule))
    return out


def group_stats(rewards, G):
    """Share of question groups (G consecutive rollouts) with no reward variance, which give GRPO no gradient."""
    groups = [rewards[g:g + G] for g in range(0, len(rewards), G)]
    flat = [max(x) - min(x) < 1e-8 for x in groups]
    n = max(len(groups), 1)
    return {"zero_var_share": sum(flat) / n,
            "zero_var_all_zero": sum(f and abs(x[0]) < 1e-8 for f, x in zip(flat, groups)) / n,
            "flat_groups": [i for i, f in enumerate(flat) if f]}


class WarmupTrigger:
    """Self triggered warm up: w(t) = 0 until the moving average of training reward improves by
    less than eps over `window` steps, then w rises linearly to 1 over `ramp` steps.

    Defaults: a 20 step mean that gains less than 0.01 over 40 steps. The interim report's rule (10 step mean,
    under 0.005 over 20 steps) is below the noise of 64 rollouts per step and fires while reward still rises: it
    did so at step 36 of a stage 0 sweep run. OLD_TRIGGER keeps it for the comparison in the report."""

    def __init__(self, window=40, eps=0.01, ramp=50, ma=20):
        self.window, self.eps, self.ramp, self.ma = window, eps, ramp, ma
        self.rewards, self.mas, self.trigger_step = [], [], None

    def update(self, reward_mean):
        self.rewards.append(reward_mean)
        recent = self.rewards[-self.ma:]
        self.mas.append(sum(recent) / len(recent))
        t = len(self.rewards) - 1
        if self.trigger_step is None and len(self.mas) > self.window + self.ma:
            if self.mas[-1] - self.mas[-1 - self.window] < self.eps:
                self.trigger_step = t
        return self.weight(t)

    def weight(self, t):
        if self.trigger_step is None:
            return 0.0
        return max(0.0, min(1.0, (t - self.trigger_step) / self.ramp))


OLD_TRIGGER = dict(window=20, eps=0.005, ma=10)


def replay_trigger(rewards, **kw):
    """Step (1-based, as in log.jsonl) after which the trigger fires on a logged reward curve, or None."""
    t = WarmupTrigger(**kw)
    for r in rewards:
        t.update(r)
    return None if t.trigger_step is None else t.trigger_step + 1


class Warmup:
    """w(t) per optimizer step, keyed by the trainer's global_step.

    spec  "off"            w = 1 from the first step (the penalty is simply on, e.g. a run branched at the switch)
          "manual_step:N"  w = 0 before step N, then a linear ramp to 1 over `ramp` steps (ramp 0 = SAAS's hard switch)
          "self"           WarmupTrigger on the training reward
    observe() may be called many times within one optimizer step (reward calls, accumulation micro-batches,
    resampling); a step's mean reward is committed to the trigger exactly once, when a later step asks for its weight.
    The trigger is tracked for every spec, so any run's log shows when the self trigger would have fired."""

    def __init__(self, spec="off", ramp=None, **trigger_kw):
        self.spec = spec
        self.manual = int(spec.split(":", 1)[1]) if spec.startswith("manual_step:") else None
        if spec not in ("off", "self") and self.manual is None:
            raise ValueError(f"unknown warm up spec {spec!r}")
        self.ramp = (50 if spec == "self" else 0) if ramp is None else ramp
        self.trigger = WarmupTrigger(ramp=max(self.ramp, 1), **trigger_kw)
        self.pending = {}  # step -> [reward sum, count], not yet committed

    @property
    def trigger_step(self):
        return self.trigger.trigger_step

    def observe(self, step, reward_mean):
        s = self.pending.setdefault(step, [0.0, 0])
        s[0] += reward_mean
        s[1] += 1

    def _commit(self, upto):
        while len(self.trigger.rewards) < upto:
            t = len(self.trigger.rewards)
            if t in self.pending:
                total, count = self.pending.pop(t)
                self.trigger.update(total / count)
            else:  # a step without its own rollouts reuses the last reward
                self.trigger.update(self.trigger.rewards[-1] if self.trigger.rewards else 0.0)

    def weight(self, step):
        self._commit(step)  # w(step) only depends on the rewards of earlier steps
        if self.spec == "off":
            return 1.0
        if self.manual is not None:
            if step < self.manual:
                return 0.0
            return 1.0 if self.ramp == 0 else min(1.0, (step - self.manual) / self.ramp)
        return self.trigger.weight(step)

    def state(self):
        rewards = list(self.trigger.rewards)
        while len(rewards) in self.pending:  # steps observed but not yet committed
            total, count = self.pending[len(rewards)]
            rewards.append(total / count)
        return {"spec": self.spec, "ramp": self.ramp, "rewards": rewards, "trigger_step": self.trigger_step}

    def restore(self, state, upto):
        """Replay the saved reward history of steps < upto (the step training resumes from)."""
        self.trigger.rewards, self.trigger.mas, self.trigger.trigger_step = [], [], None
        self.pending = {}
        for r in state["rewards"][:upto]:
            self.trigger.update(r)