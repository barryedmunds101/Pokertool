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
- `showdown.py` compares two private hands on a completed river board.
- `tests/` contains the automated pytest suite.
- `experiments/` is reserved for exploratory notebooks.

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
from showdown import compare_private_hands

first = make_hand(["Jc", "Ad"])
second = make_hand(["9c", "Kd"])
river = make_hand(["2c", "7d", "9h", "Js", "3c"])

result = compare_private_hands(first, second, river)
print(result)  # 2: the first hand wins
```

Overlapping private or board cards raise `ValueError` instead of producing an
invalid result.

## Tests

Install the development dependency and run the suite from the repository root:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

The notebooks in `tests/` remain as interactive demonstrations. The `.py`
tests are the authoritative automated test suite.
