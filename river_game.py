"""A small, exact heads-up river extensive-form game."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TypeAlias

from compatibility import compatible
from showdown import LOSS, TIE, WIN, _validate_cards, compare_hands


CHECK = "check"
BET = "bet"
FOLD = "fold"
CALL = "call"
Action: TypeAlias = str


@dataclass(frozen=True)
class RiverGameState:
    """One underlying state in a single-bet heads-up river game.

    ``pot`` includes all chips currently in the pot. ``contributions`` records
    only chips added after this river game began, allowing terminal utility to
    be measured relative to the players' stacks at the start of the spot.
    """

    board: int
    hands: tuple[int, int]
    player_to_act: int | None
    stacks: tuple[int, int]
    contributions: tuple[int, int]
    pot: int
    bet_size: int
    action_history: tuple[Action, ...] = ()
    fold_winner: int | None = None

    def __post_init__(self) -> None:
        _validate_cards(self.board, 5, "board")
        _validate_cards(self.hands[0], 2, "hands[0]")
        _validate_cards(self.hands[1], 2, "hands[1]")
        if not compatible(self.hands[0], self.hands[1]):
            raise ValueError("the private hands overlap")
        if not compatible(self.board, self.hands[0]) or not compatible(
            self.board, self.hands[1]
        ):
            raise ValueError("a private hand overlaps the board")
        if self.player_to_act not in (0, 1, None):
            raise ValueError("player_to_act must be 0, 1, or None")
        if len(self.stacks) != 2 or min(self.stacks) < 0:
            raise ValueError("stacks must contain two non-negative integers")
        if len(self.contributions) != 2 or min(self.contributions) < 0:
            raise ValueError("contributions must contain two non-negative integers")
        if self.pot < 0:
            raise ValueError("pot must be non-negative")
        if self.bet_size <= 0:
            raise ValueError("bet_size must be positive")
        if self.fold_winner is not None and self.fold_winner not in (0, 1):
            raise ValueError("fold_winner must be 0, 1, or None")
        if self.terminal != (self.player_to_act is None):
            raise ValueError("terminal states must not have a player to act")

    @property
    def terminal(self) -> bool:
        """Whether betting is over; distinct from a river chance state."""
        return self.fold_winner is not None or self.action_history in {
            (CHECK, CHECK),
            (BET, CALL),
            (CHECK, BET, CALL),
        }


@dataclass(frozen=True)
class InformationSetKey:
    """Everything the acting player observes, excluding the opponent's hand."""

    player: int
    own_hand: int
    board: int
    action_history: tuple[Action, ...]
    stacks: tuple[int, int]
    contributions: tuple[int, int]
    pot: int
    bet_size: int


@dataclass(frozen=True)
class DealEvaluation:
    first: int
    second: int
    probability: Fraction
    first_player_utility: Fraction
    second_player_utility: Fraction


@dataclass(frozen=True)
class StrategyEvaluation:
    deals: tuple[DealEvaluation, ...]
    first_player_ev: Fraction
    second_player_ev: Fraction


Policy: TypeAlias = Callable[
    [InformationSetKey, tuple[Action, ...]], Mapping[Action, int | Fraction | float]
]


def new_river_game(
    board: int,
    first: int,
    second: int,
    *,
    pot: int = 2,
    stacks: tuple[int, int] = (10, 10),
    bet_size: int = 1,
    first_player: int = 0,
) -> RiverGameState:
    """Create the root of the one-bet river game."""
    if first_player not in (0, 1):
        raise ValueError("first_player must be 0 or 1")
    if min(stacks) < bet_size:
        raise ValueError("both players need at least one full bet")
    return RiverGameState(
        board,
        (first, second),
        first_player,
        stacks,
        (0, 0),
        pot,
        bet_size,
    )


def legal_actions(state: RiverGameState) -> tuple[Action, ...]:
    """Return legal actions at this game state."""
    if state.terminal:
        return ()
    if state.action_history in ((), (CHECK,)):
        return (CHECK, BET)
    if state.action_history in ((BET,), (CHECK, BET)):
        return (FOLD, CALL)
    raise ValueError(f"unrecognised action history: {state.action_history!r}")


def information_set_key(state: RiverGameState) -> InformationSetKey:
    """Return the acting player's observation, never the opponent's hand."""
    if state.terminal or state.player_to_act is None:
        raise ValueError("terminal states do not have an information set")
    player = state.player_to_act
    return InformationSetKey(
        player,
        state.hands[player],
        state.board,
        state.action_history,
        state.stacks,
        state.contributions,
        state.pot,
        state.bet_size,
    )


def apply_action(state: RiverGameState, action: Action) -> RiverGameState:
    """Apply one legal action and return the resulting immutable state."""
    actions = legal_actions(state)
    if action not in actions:
        raise ValueError(f"{action!r} is not legal; expected one of {actions}")
    assert state.player_to_act is not None
    player = state.player_to_act
    opponent = 1 - player
    history = state.action_history + (action,)

    if action == CHECK:
        if history == (CHECK, CHECK):
            return replace(state, player_to_act=None, action_history=history)
        return replace(state, player_to_act=opponent, action_history=history)

    if action == BET:
        stacks = list(state.stacks)
        contributions = list(state.contributions)
        stacks[player] -= state.bet_size
        contributions[player] += state.bet_size
        return replace(
            state,
            player_to_act=opponent,
            stacks=tuple(stacks),
            contributions=tuple(contributions),
            pot=state.pot + state.bet_size,
            action_history=history,
        )

    if action == FOLD:
        return replace(
            state,
            player_to_act=None,
            action_history=history,
            fold_winner=opponent,
        )

    amount = state.contributions[opponent] - state.contributions[player]
    if amount <= 0 or amount > state.stacks[player]:
        raise ValueError("the player cannot make the required call")
    stacks = list(state.stacks)
    contributions = list(state.contributions)
    stacks[player] -= amount
    contributions[player] += amount
    return replace(
        state,
        player_to_act=None,
        stacks=tuple(stacks),
        contributions=tuple(contributions),
        pot=state.pot + amount,
        action_history=history,
    )


def terminal_utility(state: RiverGameState, player: int = 0) -> Fraction:
    """Return a player's chip gain relative to the start of this river spot."""
    if not state.terminal:
        raise ValueError("utility is only defined for terminal game states")
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    if state.fold_winner is not None:
        share = Fraction(state.pot if state.fold_winner == player else 0)
    else:
        result = compare_hands(state.hands[player], state.hands[1 - player], state.board)
        if result == WIN:
            share = Fraction(state.pot)
        elif result == LOSS:
            share = Fraction()
        else:
            share = Fraction(state.pot, 2)
    return share - state.contributions[player]


def _normalise_policy(
    probabilities: Mapping[Action, int | Fraction | float],
    actions: tuple[Action, ...],
) -> tuple[tuple[Action, Fraction], ...]:
    if set(probabilities) != set(actions):
        raise ValueError(f"strategy must assign exactly the legal actions {actions}")
    result = tuple((action, Fraction(probabilities[action])) for action in actions)
    if any(probability < 0 for _, probability in result):
        raise ValueError("strategy probabilities cannot be negative")
    if sum((probability for _, probability in result), Fraction()) != 1:
        raise ValueError("strategy probabilities must sum exactly to one")
    return result


def strategy_utility(
    state: RiverGameState,
    strategies: tuple[Policy, Policy],
    player: int = 0,
) -> Fraction:
    """Evaluate two behavioural strategies recursively from one fixed deal."""
    if state.terminal:
        return terminal_utility(state, player)
    assert state.player_to_act is not None
    actions = legal_actions(state)
    key = information_set_key(state)
    policy = _normalise_policy(strategies[state.player_to_act](key, actions), actions)
    return sum(
        probability * strategy_utility(apply_action(state, action), strategies, player)
        for action, probability in policy
    )


def compatible_deals(
    first_range: Mapping[int, int | Fraction] | Sequence[int],
    second_range: Mapping[int, int | Fraction] | Sequence[int],
    board: int,
) -> tuple[tuple[int, int, Fraction], ...]:
    """Return compatible deals with blocker-conditioned product probabilities."""
    _validate_cards(board, 5, "board")
    first_weights = _range_weights(first_range, "first_range")
    second_weights = _range_weights(second_range, "second_range")
    weighted: list[tuple[int, int, Fraction]] = []
    for first, first_weight in first_weights.items():
        for second, second_weight in second_weights.items():
            if not (first & second or first & board or second & board):
                weighted.append((first, second, first_weight * second_weight))
    total = sum((weight for _, _, weight in weighted), Fraction())
    if not total:
        raise ValueError("the ranges contain no compatible private-hand deals")
    return tuple((first, second, weight / total) for first, second, weight in weighted)


def _range_weights(
    value: Mapping[int, int | Fraction] | Sequence[int], name: str
) -> dict[int, Fraction]:
    items = value.items() if isinstance(value, Mapping) else ((hand, 1) for hand in value)
    result: dict[int, Fraction] = {}
    for hand, weight in items:
        _validate_cards(hand, 2, name)
        exact_weight = Fraction(weight)
        if exact_weight <= 0:
            raise ValueError(f"{name} weights must be positive")
        result[hand] = exact_weight
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def evaluate_strategies(
    board: int,
    first_range: Mapping[int, int | Fraction] | Sequence[int],
    second_range: Mapping[int, int | Fraction] | Sequence[int],
    strategies: tuple[Policy, Policy],
    *,
    pot: int = 2,
    stacks: tuple[int, int] = (10, 10),
    bet_size: int = 1,
    first_player: int = 0,
) -> StrategyEvaluation:
    """Evaluate strategies exactly over all compatible weighted deals."""
    deals: list[DealEvaluation] = []
    for first, second, probability in compatible_deals(first_range, second_range, board):
        state = new_river_game(
            board,
            first,
            second,
            pot=pot,
            stacks=stacks,
            bet_size=bet_size,
            first_player=first_player,
        )
        first_utility = strategy_utility(state, strategies, 0)
        second_utility = strategy_utility(state, strategies, 1)
        deals.append(
            DealEvaluation(first, second, probability, first_utility, second_utility)
        )
    result = tuple(deals)
    return StrategyEvaluation(
        result,
        sum((deal.probability * deal.first_player_utility for deal in result), Fraction()),
        sum((deal.probability * deal.second_player_utility for deal in result), Fraction()),
    )
