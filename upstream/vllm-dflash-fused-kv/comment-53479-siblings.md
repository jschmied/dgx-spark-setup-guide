Ran the two sibling pairs. Neither prediction holds on this build — parent 6 400 tokens, sibling
shares N leading tokens and diverges, each pair from its **own** corpus region:

| pair | shared | predicted base → PR | measured base | measured PR |
|---|---:|---:|---:|---:|
| A | 3 400 | 0 → 1 600 | **0** | **0** |
| B | 5 000 | 0 → 3 200 | **3 200** | **3 200** |

So the sibling that should gain does not move, and the one that should start at zero already hits
on base. `SCHEDPROBE` in each arm confirms `last_cache_position` 4 800 (base) against 6 400 (PR),
and the extra chunk stop at 1 600 in the PR arm.

One trap worth naming: my first attempt shared one parent across both pairs, which lets pair A's
sibling cache blocks pair B's sibling then hits — a base-line hit that is not real. Giving each pair
its own material changed nothing, so that was not the cause here.

Caveat unchanged: this build predates #52789, so the PR is applied as its intent for the
no-checkpoint case rather than verbatim. The offer to re-run against your branch on a
#52789-inclusive build stands.
