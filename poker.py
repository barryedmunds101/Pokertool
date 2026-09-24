"""Texas Hold'em boards and private hands using card bitsets."""

from __future__ import annotations

from dataclasses import dataclass

from cards import Deck, FULL_DECK, hand_names


@dataclass(frozen=True)
class HoldemDeal:
    """A completed Texas Hold'em deal.

    Every hand and board component is stored as the same integer bitset used
    by :mod:`cards`.
    """

    private_hands: tuple[int, ...]
    flop: int
    turn: int
    river: int
    burn_cards: int = 0

    def __post_init__(self) -> None:
        if not 2 <= len(self.private_hands) <= 10:
            raise ValueError("Texas Hold'em requires between 2 and 10 players")

        groups = (*self.private_hands, self.flop, self.turn, self.river, self.burn_cards)
        expected_counts = (2,) * len(self.private_hands) + (3, 1, 1)
        if self.burn_cards:
            expected_counts += (3,)
        else:
            expected_counts += (0,)

        used = 0
        for group, expected in zip(groups, expected_counts):
            if group < 0 or group & ~FULL_DECK:
                raise ValueError("A deal may only contain cards from the 52-card deck")
            if group.bit_count() != expected:
                raise ValueError(f"Expected {expected} cards, got {group.bit_count()}")
            if used & group:
                raise ValueError("The same card appears more than once in the deal")
            used |= group

    @property
    def board(self) -> int:
        """Return the complete five-card community board."""
        return self.flop | self.turn | self.river

    @property
    def all_dealt_cards(self) -> int:
        """Return private, board, and burn cards as one bitset."""
        dealt = self.board | self.burn_cards
        for hand in self.private_hands:
            dealt |= hand
        return dealt

    def player_hand(self, player: int) -> int:
        """Return a player's two private cards, using a zero-based index."""
        if player < 0:
            raise IndexError(f"No player at index {player}")
        try:
            return self.private_hands[player]
        except IndexError as error:
            raise IndexError(f"No player at index {player}") from error

    def player_cards(self, player: int) -> int:
        """Return a player's seven available cards (private cards plus board)."""
        return self.player_hand(player) | self.board


def deal_holdem(
    player_count: int = 2,
    *,
    deck: Deck | None = None,
    burn: bool = True,
) -> HoldemDeal:
    """Deal private hands and a complete Texas Hold'em board.

    Private cards are dealt one at a time around the table.  When ``burn`` is
    true, one card is burned before the flop, turn, and river.
    """
    if not isinstance(player_count, int) or isinstance(player_count, bool):
        raise TypeError("player_count must be an integer")
    if not 2 <= player_count <= 10:
        raise ValueError("player_count must be between 2 and 10")

    deck = deck if deck is not None else Deck()
    required = player_count * 2 + 5 + (3 if burn else 0)
    if len(deck) < required:
        raise ValueError(f"A deal needs {required} cards; the deck has {len(deck)}")

    hands = [0] * player_count
    for _ in range(2):
        for player in range(player_count):
            hands[player] |= deck.draw()

    burned = 0
    if burn:
        burned |= deck.draw()
    flop = deck.draw(3)
    if burn:
        burned |= deck.draw()
    turn = deck.draw()
    if burn:
        burned |= deck.draw()
    river = deck.draw()

    return HoldemDeal(tuple(hands), flop, turn, river, burned)


def describe_deal(deal: HoldemDeal) -> str:
    """Return a readable multi-line summary of a deal."""
    lines = [
        *(f"Player {number}: {' '.join(hand_names(hand))}"
          for number, hand in enumerate(deal.private_hands, start=1)),
        f"Flop: {' '.join(hand_names(deal.flop))}",
        f"Turn: {' '.join(hand_names(deal.turn))}",
        f"River: {' '.join(hand_names(deal.river))}",
        f"Board: {' '.join(hand_names(deal.board))}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_deal(deal_holdem(6)))
