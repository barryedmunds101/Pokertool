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
