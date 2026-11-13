"""Gold-free stopping controllers; all inputs describe revealed candidates only."""
from dataclasses import dataclass, asdict
from collections import Counter
import numpy as np
from atml_pa3.task3.selection import Candidate, select, descending_average_ranks
from atml_pa3.task3.equivalence import NO_ANSWER

CHECKPOINTS = (4, 8, 16)
TAUS = (.5, .625, .75, .875, 1.)
GAMMAS = (0., .1, .25, .5)

@dataclass(frozen=True)
class Policy:
    family: str
    tau: float | None = None
    gamma: float | None = None
    fixed: int | None = None

    @property
    def id(self):
        if self.family == 'fixed': return f'Fixed{self.fixed}'
        if self.family == 'always32': return 'Always32'
        return f'{self.family}:tau={self.tau:g}' + (f':gamma={self.gamma:g}' if self.gamma is not None else '')

def policies():
    return ([Policy('agreement', t) for t in TAUS] +
            [Policy('quality', t, g) for t in TAUS for g in GAMMAS] + [Policy('always32')])

def evidence(candidates, quality):
    """No correctness, oracle, gold or future scores accepted by this interface."""
    n = len(candidates)
    sc = select(candidates, 'consistency')
    rwc = select(candidates, 'rwc', quality)
    votes = Counter(c.answer_group for c in candidates)
    largest = max(votes.values())
    unique = sum(v == largest for v in votes.values()) == 1
    weights = np.rint(2 * (n - descending_average_ranks(quality))).astype(int) if n > 1 else np.ones(1, int)
    weighted = dict.fromkeys(votes, 0)
    for c, w in zip(candidates, weights): weighted[c.answer_group] += int(w)
    ranked = sorted(weighted.values(), reverse=True)
    total = sum(ranked)
    margin = (ranked[0] - (ranked[1] if len(ranked) > 1 else 0)) / total if total else 0.
    return {'n': n, 'selected_rollout_id': candidates[sc.index].rollout_id,
            'answer_group': sc.answer_group, 'leader_votes': largest,
            'leader_support': largest / n, 'unique_leader': unique,
            'no_answer': sc.answer_group == NO_ANSWER,
            'rwc_group': rwc.answer_group, 'rwc_agrees': rwc.answer_group == sc.answer_group,
            'rwc_margin': margin, 'votes': dict(votes), 'rwc_vote_numerators': weighted,
            'supporting_ids': [c.rollout_id for c in candidates if c.answer_group == sc.answer_group],
            'consumed_ids': [c.rollout_id for c in candidates], 'token_cost': sc.token_cost}

def should_stop(policy, snapshot):
    if snapshot['n'] not in CHECKPOINTS: return False
    agreement = (snapshot['unique_leader'] and not snapshot['no_answer'] and
                 snapshot['leader_support'] >= policy.tau) if policy.tau is not None else False
    if policy.family == 'agreement': return agreement
    if policy.family == 'quality': return agreement and snapshot['rwc_agrees'] and snapshot['rwc_margin'] >= policy.gamma
    return False

def decide(policy, snapshots):
    if policy.family == 'fixed': return snapshots[policy.fixed]
    for n in CHECKPOINTS:
        if should_stop(policy, snapshots[n]): return snapshots[n]
    return snapshots[32]

def choose_policy(grid, family, fixed32_accuracy):
    eligible = [r for r in grid if r['family'] in (family, 'always32') and
                r['accuracy'] >= fixed32_accuracy - .01 - 1e-12]
    if not eligible: raise ValueError('Missing always-32 fallback')
    return min(eligible, key=lambda r: (r['token_cost'], -r['accuracy'],
               r['tau'] if r['tau'] is not None else float('inf'),
               r['gamma'] if r['gamma'] is not None else float('inf')))

def policy_dict(p): return {**asdict(p), 'policy_id': p.id}

def from_dict(row): return Policy(**{k: row.get(k) for k in ('family', 'tau', 'gamma', 'fixed')})
