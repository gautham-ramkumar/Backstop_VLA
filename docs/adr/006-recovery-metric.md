# ADR-006: A recovery is judged on the paired change in success rate, with its cost beside it

- Status: Accepted (2026-11-04)
- Date: 2026-11-04
- Amends: nothing. ADR-001 and ADR-005 still govern the guard.

## Context

Until now the guard only stopped runs, and a success rate was banned as its
headline (ADR-001, ADR-005): a guard that never fires keeps the policy's
success rate, so the number says nothing about the guard.

Recovery changes what the guard is for. When it fires, the arm lifts and
backs off and the policy tries again (`docs/recovery.md`,
`backstop.recover.controller`). The question is now whether more runs
succeed, so a success rate is the honest number. On its own it would still
mislead in two ways:

- **It hides the cost.** Retries take time, and a guard that fires on a good
  run can turn a success into a failure. A higher success rate bought with
  doubled episode lengths and spoiled good runs is a different result from
  one bought cheaply.
- **Unpaired, it is mostly noise.** Two separate batches of 100 episodes
  differ by several points by chance. Run from the same start states, the
  only episodes that can move the number are the ones the two arms disagree
  on.

## Decision

1. **Headline: the paired change in success rate** (`paired_success_rate_change`):
   the guard-on success rate minus the guard-off one, on the same episodes from
   the same start states. Branching from the recorded sim state makes the
   pairing exact (`backstop.record.recovery`). Episodes the guard never fires on
   run as recorded in both arms.
2. **Always beside it** (`schema.RECOVERY_ALONGSIDE`; `write_recovery_report`
   refuses a report without them):
   - `n_paired`: how many episodes the comparison covers;
   - `rescued` and `spoiled`: failures turned into successes, and successes
     turned into failures. The headline is their difference over `n_paired`;
   - `p_value`: McNemar's exact test on those two counts;
   - `intervention_rate`: the share of episodes the guard stepped into;
   - `max_attempts`: the attempt budget, which is part of the result. Unbounded
     retry would turn failures into runs that never end;
   - `mean_extra_steps`: the time cost, guard-on length minus guard-off length,
     over the episodes the guard stepped into.
3. **The guard itself is unchanged.** The frozen rules
   (`artifacts/guard/frozen/rule_{05,20}.json`) fire exactly as ADR-005's
   headline guard does. After the attempt budget is spent they halt the run, as
   they did before recovery existed. ADR-005's headline and the ban on a success
   rate as a guard headline stand.

## Alternatives considered

- **Guard-on success rate alone.** Rejected for the two reasons above.
- **Unpaired batches.** Rejected: twice the GPU time for a noisier answer, when
  branching from the recorded state gives the exact pairing for free.
- **Time to success.** A useful secondary number, but a policy that recovers
  slowly and one that never recovers would both look bad on it, for different
  reasons.
