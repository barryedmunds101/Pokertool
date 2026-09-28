from fractions import Fraction

import pytest

from cards import make_hand
from river_game import (
    BET,
    CALL,
    CHECK,
    FOLD,
    apply_action,
    compatible_deals,
    evaluate_strategies,
    information_set_key,
    legal_actions,
    new_river_game,
    terminal_utility,
)


def hand(*names: str) -> int:
    return make_hand(names)


BOARD = hand("2c", "7d", "9h", "Js", "3c")
FIRST_STRONG = hand("Jc", "Ad")
FIRST_WEAK = hand("4d", "5d")
SECOND_STRONG = hand("9c", "Kd")
SECOND_WEAK = hand("8c", "Qd")


def pure(action):
    return lambda key, actions: {
        candidate: Fraction(candidate == action) for candidate in actions
    }


def test_complete_one_bet_action_tree() -> None:
    root = new_river_game(BOARD, FIRST_STRONG, SECOND_STRONG)
    assert not root.terminal
    assert legal_actions(root) == (CHECK, BET)

    after_check = apply_action(root, CHECK)
    assert legal_actions(after_check) == (CHECK, BET)
    checked_down = apply_action(after_check, CHECK)
    assert checked_down.terminal
    assert legal_actions(checked_down) == ()

    after_bet = apply_action(root, BET)
    assert after_bet.pot == 3
    assert after_bet.stacks == (9, 10)
    assert after_bet.contributions == (1, 0)
    assert legal_actions(after_bet) == (FOLD, CALL)

    folded = apply_action(after_bet, FOLD)
    called = apply_action(after_bet, CALL)
    assert folded.terminal and called.terminal
    assert called.pot == 4
    assert called.contributions == (1, 1)


def test_information_set_excludes_opponents_private_hand() -> None:
    first_state = new_river_game(BOARD, FIRST_STRONG, SECOND_STRONG)
    second_state = new_river_game(BOARD, FIRST_STRONG, SECOND_WEAK)

    assert information_set_key(first_state) == information_set_key(second_state)
    assert SECOND_STRONG not in information_set_key(first_state).__dict__.values()
    assert SECOND_WEAK not in information_set_key(second_state).__dict__.values()

    different_own_hand = new_river_game(BOARD, FIRST_WEAK, SECOND_STRONG)
    assert information_set_key(first_state) != information_set_key(different_own_hand)


def test_terminal_chip_payoffs_for_showdown_and_fold() -> None:
    root = new_river_game(BOARD, FIRST_STRONG, SECOND_STRONG)
    called = apply_action(apply_action(root, BET), CALL)
    folded = apply_action(apply_action(root, BET), FOLD)
    checked_down = apply_action(apply_action(root, CHECK), CHECK)

    assert (terminal_utility(called, 0), terminal_utility(called, 1)) == (3, -1)
    assert (terminal_utility(folded, 0), terminal_utility(folded, 1)) == (2, 0)
    assert (terminal_utility(checked_down, 0), terminal_utility(checked_down, 1)) == (2, 0)
    assert terminal_utility(called, 0) + terminal_utility(called, 1) == 2


def test_compatible_deals_are_blocker_conditioned() -> None:
    blocked = hand("2c", "Ah")
    deals = compatible_deals(
        [FIRST_STRONG, FIRST_WEAK, blocked],
        [SECOND_STRONG, SECOND_WEAK],
        BOARD,
    )

    assert len(deals) == 4
    assert all(probability == Fraction(1, 4) for _, _, probability in deals)
    assert sum((probability for _, _, probability in deals), Fraction()) == 1


def test_two_supplied_strategies_have_hand_checkable_exact_ev() -> None:
    def first_policy(key, actions):
        if actions == (CHECK, BET):
            choice = BET if key.own_hand == FIRST_STRONG else CHECK
        else:
            choice = FOLD
        return {action: Fraction(action == choice) for action in actions}

    def second_calls(key, actions):
        choice = CALL if actions == (FOLD, CALL) else CHECK
        return {action: Fraction(action == choice) for action in actions}

    def second_folds(key, actions):
        choice = FOLD if actions == (FOLD, CALL) else CHECK
        return {action: Fraction(action == choice) for action in actions}

    ranges = ([FIRST_STRONG, FIRST_WEAK], [SECOND_STRONG, SECOND_WEAK])
    call_result = evaluate_strategies(
        BOARD, ranges[0], ranges[1], (first_policy, second_calls)
    )
    fold_result = evaluate_strategies(
        BOARD, ranges[0], ranges[1], (first_policy, second_folds)
    )

    assert len(call_result.deals) == len(fold_result.deals) == 4
    assert call_result.first_player_ev == Fraction(3, 2)
    assert call_result.second_player_ev == Fraction(1, 2)
    assert fold_result.first_player_ev == 1
    assert fold_result.second_player_ev == 1
    assert call_result.first_player_ev + call_result.second_player_ev == 2


def test_illegal_actions_and_invalid_strategy_probabilities_are_rejected() -> None:
    root = new_river_game(BOARD, FIRST_STRONG, SECOND_STRONG)
    with pytest.raises(ValueError, match="not legal"):
        apply_action(root, CALL)

    bad_policy = lambda key, actions: {action: Fraction(1, 3) for action in actions}
    with pytest.raises(ValueError, match="sum exactly"):
        evaluate_strategies(
            BOARD,
            [FIRST_STRONG],
            [SECOND_STRONG],
            (bad_policy, pure(CHECK)),
        )
