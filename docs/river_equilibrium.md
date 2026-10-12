# One small river equilibrium

This extension connects the existing chance/mass model and river betting tree
to an exact equilibrium, an independent certificate, and decision explanations.
It is a small reference against which future sequence-form or CFR solvers can
be checked. No new runtime dependency is required.

## Run it

From the repository root, with the project's Python environment activated:

```powershell
python -m experiments.river_equilibrium
python -m experiments.river_equilibrium --verify examples/polar_river_certificate.json
python -m pytest -q
```

To save another copy of the benchmark's certificate:

```powershell
python -m experiments.river_equilibrium --certificate my_river_certificate.json
python -m experiments.river_equilibrium --verify my_river_certificate.json
```

The `--verify` command runs the checker, not the solver. A failed check exits
with an error instead of reporting a tolerance-based or approximate success.

## The first spot

| Setting | Value |
| --- | --- |
| Board | 2c 7d 9h Js 3c |
| Player 0 hands | Jc Ad and 4d 5d, equal starting mass |
| Player 1 hand | 9c Kd |
| Starting pot | 1 blind |
| Fixed bet | 1 blind |
| First to act | Player 0 |
| Betting tree | check/check; bet/fold or bet/call; check/bet/fold or check/bet/call |

All deals pass the existing private/board-card compatibility checks. The
first player knows their hand, while the second player does not distinguish
the two underlying deals. Decisions are tied by the existing information-set
keys, so the defender cannot call only against the bluff by observing a hidden
hand.

The LP selects an equilibrium in which Jc Ad bets with probability 1, 4d 5d
bets with probability 1/2, and 9c Kd calls an initial bet with probability 1/2.
The tree also specifies the decisions following an initial check; consult the
[full report](polar_river_report.md) for all six information sets.

After observing an initial bet, the defender's opponent-hand probabilities
are 2/3 value and 1/3 bluff. These are posterior masses conditioned on the
betting strategy, not the original 1/2 and 1/2 range weights. A pot-sized
river call breaks even at equity 1/3, so calling and folding are exactly tied.
The bluff is also indifferent: against a 1/2 call frequency, its fold-relative
bet EV is `(1/2)*1 - (1/2)*1 = 0`, matching a check.

Equilibrium need not be unique. A pure action can be selected even when another
action has the same continuation value. The report labels unreachable
information sets rather than treating a Nash certificate as a proof of
sequential optimality at every off-path history.

## Utility convention

The existing `terminal_utility` gives chip gain relative to the river's initial
stacks. Its two player utilities sum to the already-present starting pot.
The solver subtracts half the starting pot from each player's utility. This
produces a zero-sum payoff without changing decisions or best responses:

`centered_utility(player) = terminal_utility(player) - initial_pot/2`.

For this spot, centered player-0 value is exactly 1/4 blind. The original chip
gains are 3/4 blind for player 0 and 1/4 blind for player 1. Both quantities are
shown in the report to avoid confusing a convention change with a poker EV
improvement.

## How the algorithm works

1. `RiverSpot` accepts a five-card board and exact `JointMass`. Its
   `from_ranges` constructor uses `compatible_joint_mass`, removing blockers
   and normalizing the compatible product mass. Correlated joint mass can also
   be supplied directly.
2. `compile_river_spot` walks the existing river tree. It records legal actions
   and each information set's own predecessor sequence, checking perfect
   recall. It enumerates deterministic complete plans: 16 for player 0 and 4
   for player 1 in the benchmark.
3. Each matrix entry is the chance-weighted centered terminal payoff of one
   pair of plans. Only these finite pure plans are enumerated; fractions and
   mixed strategy profiles are never enumerated.
4. `solve_zero_sum` solves a positive-shifted packing LP using `Fraction`
   arithmetic and Bland's simplex pivot rule. It returns rational mixtures and
   checks all matrix saddle inequalities.
5. `mixture_to_behavior` conditions each local action on the player's own
   predecessor realization mass. Unweighted averaging of local actions would
   generally change the strategy. Unused own continuations receive an optimal
   counterfactual action when an opponent-conditioned value is available.
6. `certify_profile` evaluates the recovered profile and every pure unilateral
   deviation through the original `strategy_utility`. It ignores the LP's
   matrix and tableau.

The small normal form is intentionally inspectable. Its default limits are 32
compatible deals, 256 pure plans per player and 65,536 matrix entries. A limit
failure asks for a larger-game representation; it does not silently discard
hands, histories or actions. Exact simplex also has a pivot limit and returns
no certified answer if that limit is reached.

## What the certificate proves

Write `U` for centered player-0 payoff and `(x,y)` for the stated profile. The
checker computes these finite quantities independently:

- `value = U(x,y)`;
- `lower_bound = min_y' U(x,y')`, with player 0 fixed;
- `upper_bound = max_x' U(x',y)`, with player 1 fixed;
- player-0 gain is `upper_bound - value`;
- player-1 gain is `value - lower_bound`;
- best-response gap is `upper_bound - lower_bound`.

Every alternative mixed plan is a weighted average of deterministic plans.
Checking every deterministic information-set plan therefore bounds every
legal deviation, including changes at several decisions together. The
benchmark's bounds both equal 1/4 and its gap is exactly zero. This is a
finite equality over rational numbers, with no numerical tolerance.

The JSON witness embeds the board, normalized joint deals, betting rules and
both full behavioural policies. Rational quantities are strings such as
`"1/2"`, avoiding rounded JSON floats. The checker rejects missing/duplicate
information sets, illegal probabilities, inconsistent claimed values and a
positive best-response gap. The certificate establishes equilibrium for its
embedded spot only. Changing ranges, bets, rules or the board requires a new
check.

## Explain decisions and extend the model

`explain_profile` aggregates underlying nodes at their information sets.
Counterfactual weights include chance and opponent reach, excluding the
acting player's own predecessor reach. The output gives actual reach,
opponent beliefs, showdown equity, action probabilities and conditional
continuation values. It does not optimize separately against hidden hands.

For a different small spot:

```python
from cards import make_hand
from mass import RangeMass
from river_equilibrium import (
    RiverSpot, compile_river_spot, solve_river_equilibrium,
    certificate_to_dict, verify_certificate, explain_profile,
)

spot = RiverSpot.from_ranges(
    make_hand(["2c", "7d", "9h", "Js", "3c"]),
    RangeMass.from_hands({make_hand(["Jc", "Ad"]): 2,
                          make_hand(["4d", "5d"]): 1}),
    RangeMass.uniform([make_hand(["9c", "Kd"])]),
    pot=1, bet_size=1,
)
game = compile_river_spot(spot)
result = solve_river_equilibrium(game)
assert verify_certificate(certificate_to_dict(game, result.profile)).gap == 0
decisions = explain_profile(game, result.profile)
```

The next larger-game step is to replace pure-plan matrix expansion with
sequence-form constraints, retaining the same tree evaluator, information-set
rules and certificate meaning. A larger solver can be compared on values and
best-response gaps; it need not reproduce the same equilibrium probabilities.
