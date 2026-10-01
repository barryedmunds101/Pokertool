# Pokertool

Pokertool is a small foundation for exact Texas Hold'em calculations. Cards
are represented by bits in a 52-bit integer: bits 0 through 51 correspond to
the 52 cards in a standard deck.

This makes overlap checks inexpensive:

```python
compatible = first_cards & second_cards == 0
```

## Modules

- `cards.py` creates cards, hands, and shuffled decks.
- `poker.py` deals private hands and Texas Hold'em boards.
- `compatibility.py` finds remaining cards and compatible private hands.
- `hand_space.py` defines the fixed 1,326-coordinate private-hand basis.
- `showdown.py` compares two private hands on a completed river board.
- `river_kernel.py` builds numeric compatibility and dominance operators.
- `turn.py` enumerates every labelled river and caches canonical showdowns.
- `flop.py` recursively aggregates every turn while sharing suit-isomorphic
  descendant calculations.
- `preflop.py` retains every legal flop summary while evaluating one
  representative per conditioned suit orbit.
- `chance.py` generates one street of exact chance children and canonical
  stabilizer orbits without evaluating descendants.
- `mass.py` represents exact private-hand ranges and joint mass, propagates
  mass through chance children, and provides weighted showdown signatures
  and optional conditional beliefs.
- `river_game.py` defines a complete one-bet heads-up river game, information
  sets, terminal chip utilities and exact strategy evaluation.
- `symmetry.py` implements the 24 suit permutations, orbits, and stabilizers.
- `tests/` contains the automated pytest suite.
- `experiments/` contains exploratory notebooks, including a guided river
  kernel and suit-symmetry experiment.

## Example

```python
from cards import hand_names, make_hand
from compatibility import compatible_private_hands
from poker import deal_holdem, describe_deal

board = make_hand(["As", "Kd", "Qh", "Jc", "2s"])
possible_hands = list(compatible_private_hands(board))
print(len(possible_hands))  # 1081

deal = deal_holdem(6)
print(describe_deal(deal))
print(hand_names(deal.board))
```

## Compare two private hands

The comparison result is from the first hand's perspective: `0` means a loss,
`1` a tie, and `2` a win.

```python
from cards import make_hand
from showdown import compare_hands

first = make_hand(["Jc", "Ad"])
second = make_hand(["9c", "Kd"])
river = make_hand(["2c", "7d", "9h", "Js", "3c"])

result = compare_hands(first, second, river)
print(result)  # 2: the first hand wins
```

Overlapping private or board cards raise `ValueError` instead of producing an
invalid result.

## Compare against every private hand

Comparison vectors and matrices use the stable 1,326-hand ordering in
`PRIVATE_HANDS`. A matrix uses that ordering for both its rows and columns.
`"I"` marks hands that overlap each other or the board.

```python
from cards import make_hand
from showdown import PRIVATE_HANDS, comparison_matrix, comparison_vector

board = make_hand(["2c", "7d", "9h", "Js", "3c"])
first = make_hand(["10c", "Ad"])

vector = comparison_vector(first, board)
matrix = comparison_matrix(board)
```

## Tests

Install the development dependency and run the suite from the repository root:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

The notebooks in `tests/` remain as interactive demonstrations. The `.py`
tests are the authoritative automated test suite.

## River chance operators

For a five-card board, `river_kernel` returns two NumPy arrays on the fixed
1,326-hand ambient basis. `C` is boolean compatibility and `D` is signed
dominance. Invalid entries are zero, `C` is symmetric, and `D` is
skew-symmetric. The doubled-equity result matrix is `R = C + D`.

```python
from cards import make_hand
from river_kernel import river_kernel

board = make_hand(["2c", "7d", "9h", "Js", "3c"])
C, D = river_kernel(board)
```

Suit permutations transport kernels equivariantly between boards. The
symmetry module derives the 169 private-hand suit classes and counts the
134,459 five-card board classes with Burnside's lemma.

## Chance children

`chance_children` exposes the non-recursive chance interface used by future
game trees. Each child contains the revealed cards, the next state and an
exact conditional probability. Optional fixed hands act as blockers.

```python
from cards import make_hand
from chance import ChanceState, chance_children

turn = make_hand(["2c", "5c", "8c", "Jc"])
first = make_hand(["Ah", "Kh"])
second = make_hand(["Js", "Jd"])

public_rivers = chance_children(ChanceState(turn))
conditioned_rivers = chance_children(ChanceState(turn, first, second))
assert len(public_rivers) == 48
assert len(conditioned_rivers) == 44
```

`canonical_chance_children` groups children only under the current state's
stabilizer. Groups retain exact multiplicities, probabilities and the suit
permutation transporting every raw member to its representative.

## Exact outcomes with symmetry reuse

The turn, flop and preflop outcome functions are consumers of the shared
chance-child interface. Symmetry changes how often a calculation is performed,
not which chance outcomes appear in the result:

| Starting state | Returned labelled children | Descendant paths represented |
| --- | ---: | ---: |
| Turn with two fixed hands | 44 rivers | 44 |
| Flop with two fixed hands | 45 turns | 1,980 ordered turn-river paths |
| Preflop with two fixed hands | 17,296 flops | 34,246,080 ordered paths |

At each state, children are partitioned by `canonical_chance_children`. The
expensive calculation is performed for one representative, then transported
or expanded back to every original child label. Exact counts and probabilities
therefore remain unchanged.

Fixed hands are part of the conditioned state. Only suit permutations that
preserve the board, each ordered private hand and recorded reveal history may
group siblings. For a generic deal this stabilizer can be trivial, so no unsafe
symmetry reduction is forced. More symmetric deals receive a larger reduction.

Terminal comparisons also use a bounded canonical cache keyed by the complete
board and both ordered hands. This reuses a showdown when the same deal is
reached through a different turn-river order or through a suit-isomorphic
representative, without merging distinct betting or reveal histories in the
chance tree.

```python
from cards import make_hand
from turn import showdown_cache_info, turn_outcomes

turn = make_hand(["2h", "3h", "2s", "3s"])
first = make_hand(["Ac", "Ad"])
second = make_hand(["Kc", "Kd"])

result = turn_outcomes(turn, first, second)
assert len(result.rivers) == 44       # every river remains available
print(showdown_cache_info())          # canonical calculations are reused
```

## Exact range and belief mass

`RangeMass` stores a non-negative integer or rational weight at every position
of the fixed 1,326-hand basis. Weights remain unnormalised until probabilities
are requested.

```python
from cards import make_hand
from mass import RangeMass

first_range = RangeMass.from_hands({
    make_hand(["Jc", "Ad"]): 2,
    make_hand(["4d", "5d"]): 1,
})
assert first_range.total == 3
```

For two marginal ranges `x` and `y`, `compatible_joint_mass` constructs the
sparse exact product model

```text
M_B(i, j) = x_i C_B(i, j) y_j.
```

Blocked hand pairs receive zero mass. `JointMass` is also public so correlated
beliefs that cannot be factored into two marginal ranges can be represented
directly. Call `to_dense()` only when the full 1,326 × 1,326 object matrix is
actually needed.

On a river, `weighted_signature` returns exact weighted wins, losses and ties.
`range_equity` agrees with the existing compatibility and dominance operators:

```text
Equity_B(x, y) = xᵀ(C_B + D_B)y / (2 xᵀC_By).
```

`turn_mass` begins with compatible pair mass on a turn and distributes every
pair equally over its 44 legal rivers. It returns all 48 public river labels,
including zero-probability rivers when the current beliefs block a card. Each
positive-probability child contains:

- its exact probability;
- its share of the parent joint mass;
- normalized conditional pair beliefs;
- exact weighted win, loss and tie mass;
- exact conditional equity.

The child masses sum exactly to the parent mass, and probability-weighted river
equity equals the aggregate turn equity.

Range weights need not respect suit symmetry. `transport_range` moves a range
through the ambient hand permutation and `range_is_invariant` tests whether a
particular suit permutation preserves it. Orbit multiplicities may be applied
to weighted calculations only when the relevant range or joint mass has the
required symmetry.

## Propagate joint mass without a belief or betting model

`propagate_joint_mass(state, joint_mass)` yields one street of
`JointMassChild` objects. Each contains `reveal`, the next `ChanceState`, and
its unnormalised `joint_mass`. Inputs may be arbitrary correlated joint mass;
there is no requirement to factor them into two ranges.

```python
from fractions import Fraction
from cards import make_hand
from chance import ChanceState
from hand_space import HAND_INDEX
from mass import JointEntry, JointMass, propagate_joint_mass

turn = make_hand(["2c", "7d", "9h", "Js"])
first = make_hand(["Jc", "Ad"])
second = make_hand(["9c", "Kd"])
parent = JointMass((JointEntry(HAND_INDEX[first], HAND_INDEX[second], 44),))

children = tuple(propagate_joint_mass(ChanceState(turn), parent))
assert len(children) == 48
assert sum(child.joint_mass.total > 0 for child in children) == 44
assert sum((child.joint_mass.total for child in children), Fraction()) == 44
# Each available river carries one unit; the four held cards carry zero.
```

The split uses each pair's legal reveals, rather than the public state's
unweighted reveal probability: a pair splits over 17,296 unordered flops,
45 turns, or 44 rivers. All child labels from `chance_children(state)` remain
present, including those carrying zero mass. Optional fixed hands in the
state restrict labels and must match every positive-mass input pair.
Incompatible input deals raise `ValueError` on iteration instead of silently
losing mass. Empty mass produces empty mass at each child. A river yields no
children; its mass remains at that terminal state.

At every nonterminal state, adding child masses recovers the parent **at each
private-hand-pair coordinate**. The operation is linear, uses exact
`Fraction` arithmetic, and preserves recorded reveal and action histories.
It performs no normalization, showdown evaluation, or descendant expansion.
Existing `turn_mass` now uses this operator before adding river probabilities
and W/L/T results, so its public interface and results remain the same.

On a flop, pass a yielded child's `state` and `joint_mass` into
`propagate_joint_mass` again to continue to the river. Branch probabilities
can be derived as `child.joint_mass.total / parent.total` when the parent has
positive mass. Call `child.joint_mass.normalized()` only when conditional
beliefs are needed and the child has positive mass.

The iterator constructs one child mass at a time; retaining all children or
using dense supports can still be expensive. Start with a few pairs. The
worked notebook [joint_mass_flow.ipynb](experiments/joint_mass_flow.ipynb)
demonstrates correlated input, blockers, coordinate-wise conservation and
repeated propagation. Suit-canonical grouping must also preserve or transport
the mass weights; board symmetry alone does not justify weighted grouping.

## Small river extensive-form game

`river_game.py` connects cards and showdowns to a complete betting tree. The
first player may check or bet. After a check, the second player may check or
bet; after a bet, the other player may fold or call. Raises are deliberately
excluded from this first game.

```python
from cards import make_hand
from river_game import BET, CALL, CHECK, FOLD, apply_action, new_river_game

board = make_hand(["2c", "7d", "9h", "Js", "3c"])
first = make_hand(["Jc", "Ad"])
second = make_hand(["9c", "Kd"])

root = new_river_game(board, first, second, pot=2, bet_size=1)
after_bet = apply_action(root, BET)
showdown = apply_action(after_bet, CALL)
assert showdown.terminal
```

The game terminal condition is separate from `ChanceState.terminal`: a river
has no further public-card children, but betting may still be active.

`information_set_key(state)` contains the acting player's own hand, public
board, public action history, stacks, contributions, pot and bet size. It never
contains the opponent's private hand. Consequently, underlying deals that look
identical to a player share the same information-set key.

`evaluate_strategies` accepts two behavioural policies and weighted or uniform
private-hand ranges. It removes incompatible deals, renormalizes their exact
`Fraction` probabilities and reports per-deal utilities and aggregate EV for
both players. Utilities measure chip gain from the beginning of the river spot;
because the starting pot is already present, the game is constant-sum rather
than zero-sum.
