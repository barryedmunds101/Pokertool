import copy
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from cards import make_hand
from experiments.river_equilibrium import polar_river_spot
from hand_space import HAND_INDEX
from mass import JointEntry, JointMass, RangeMass
from river_equilibrium import (
    RiverSpot, certificate_to_dict, certify_profile, compile_river_spot,
    evaluate_profile, explain_profile, mixture_to_behavior, plan_strategy,
    solve_river_equilibrium, verify_certificate,
)
from river_game import BET, CALL, CHECK, FOLD, evaluate_strategies


def hand(*names):
    return make_hand(names)


@pytest.fixture(scope="module")
def benchmark():
    game = compile_river_spot(polar_river_spot())
    return game, solve_river_equilibrium(game)


def test_polar_river_has_hand_checkable_equilibrium(benchmark):
    game, result = benchmark
    assert [len(infos) for infos in game.information_sets] == [4, 2]
    assert [len(plans) for plans in game.plans] == [16, 4]
    assert result.certificate.value == Fraction(1, 4)
    assert result.certificate.lower_bound == result.certificate.upper_bound == Fraction(1, 4)
    assert result.certificate.first_gain == result.certificate.second_gain == 0
    root = {key.own_hand: p for key, p in result.profile[0].items() if not key.action_history}
    assert root[hand("4d", "5d")][BET] == Fraction(1, 2)
    assert root[hand("Jc", "Ad")][BET] == 1
    response = next(p for key, p in result.profile[1].items() if key.action_history == (BET,))
    assert response[CALL] == Fraction(1, 2)


def test_behavioral_conversion_conditions_on_own_predecessor_actions(benchmark):
    game, result = benchmark
    first_plan = (CHECK, BET, FOLD, FOLD)
    second_plan = (CHECK, CHECK, FOLD, CALL)
    mixture = tuple(Fraction(1, 2) if p in (first_plan, second_plan) else Fraction() for p in game.plans[0])
    strategy = mixture_to_behavior(game, 0, mixture)
    key = next(info.key for info in game.information_sets[0]
               if info.key.own_hand == hand("Jc", "Ad") and info.key.action_history)
    assert strategy[key][CALL] == 1  # Unweighted action averaging would incorrectly give 1/2.
    expected = sum((weight * evaluate_profile(game, (plan_strategy(game, 0, plan), result.profile[1]))
                    for plan, weight in zip(game.plans[0], mixture)), Fraction())
    assert evaluate_profile(game, (strategy, result.profile[1])) == expected


def test_tree_certificate_ignores_matrix_and_optimizer(benchmark, monkeypatch):
    game, result = benchmark
    def fail(*args, **kwargs):
        raise AssertionError("the certificate checker must not invoke a solver")
    monkeypatch.setattr("river_equilibrium.solve_zero_sum", fail)
    fake = replace(game, payoff_matrix=((Fraction(999),),))
    assert certify_profile(fake, result.profile).gap == 0
    encoded = json.loads(json.dumps(certificate_to_dict(game, result.profile)))
    assert verify_certificate(encoded).value == Fraction(1, 4)


def test_profitable_profile_and_tampered_claims_are_rejected(benchmark):
    game, result = benchmark
    profile = tuple({key: dict(p) for key, p in strategy.items()} for strategy in result.profile)
    weak = next(key for key in profile[0] if key.own_hand == hand("4d", "5d") and not key.action_history)
    profile[0][weak] = {CHECK: Fraction(1), BET: Fraction()}
    assert certify_profile(game, profile).gap > 0
    with pytest.raises(ValueError, match="positive best-response gap"):
        verify_certificate(certificate_to_dict(game, profile))
    data = certificate_to_dict(game, result.profile)
    data["certificate"]["value"] = "0"
    with pytest.raises(ValueError, match="claims do not match"):
        verify_certificate(data)


def test_bad_probabilities_and_duplicate_infos_are_rejected(benchmark):
    game, result = benchmark
    data = certificate_to_dict(game, result.profile)
    bad = copy.deepcopy(data)
    bad["strategies"][0][0]["probabilities"][CHECK] = 0.5
    with pytest.raises(ValueError, match="fractions must be strings"):
        verify_certificate(bad)
    bad = copy.deepcopy(data)
    bad["strategies"][1].append(copy.deepcopy(bad["strategies"][1][0]))
    with pytest.raises(ValueError, match="duplicate"):
        verify_certificate(bad)
    bad = copy.deepcopy(data)
    bad["spot"]["deals"][0]["mass"] = "1/3"
    with pytest.raises(ValueError, match="sum exactly"):
        verify_certificate(bad)


def test_explanation_uses_betting_conditioned_opponent_mass(benchmark):
    game, result = benchmark
    rows = explain_profile(game, result.profile)
    facing_bet = next(row for row in rows if row.key.player == 1 and row.key.action_history == (BET,))
    assert facing_bet.reach == Fraction(3, 4)
    assert facing_bet.opponent_beliefs[hand("Jc", "Ad")] == Fraction(2, 3)
    assert facing_bet.opponent_beliefs[hand("4d", "5d")] == Fraction(1, 3)
    assert facing_bet.showdown_equity == Fraction(1, 3)
    assert facing_bet.action_values[CALL] == facing_bet.action_values[FOLD]
    assert "exactly equal" in facing_bet.note
    off_path = next(row for row in rows if row.key.player == 0
                    and row.key.own_hand == hand("Jc", "Ad") and row.key.action_history)
    assert off_path.reach == 0 and off_path.counterfactual_mass > 0
    assert "never reaches" in off_path.note
    assert off_path.probabilities[CALL] == 1


def test_weighted_blockers_match_original_evaluator():
    board = polar_river_spot().board
    first = RangeMass.from_hands({hand("Jc", "Ad"): 2, hand("4d", "5d"): 1, hand("2c", "Ah"): 99})
    second = RangeMass.from_hands({hand("9c", "Kd"): 3, hand("Ad", "Kc"): 1})
    game = compile_river_spot(RiverSpot.from_ranges(board, first, second, pot=3))
    assert len(game.roots) == 3  # Board overlap and the shared Ad deal were removed.
    assert sorted(p for _, p in game.roots) == [Fraction(1, 10), Fraction(3, 10), Fraction(3, 5)]
    def checks(key, actions):
        choice = CHECK if CHECK in actions else FOLD
        return {action: Fraction(action == choice) for action in actions}
    profile = tuple({info.key: checks(info.key, info.actions) for info in infos} for infos in game.information_sets)
    reference = evaluate_strategies(
        board, {hand("Jc", "Ad"): 2, hand("4d", "5d"): 1, hand("2c", "Ah"): 99},
        {hand("9c", "Kd"): 3, hand("Ad", "Kc"): 1}, (checks, checks), pot=3,
    )
    assert evaluate_profile(game, profile) == reference.first_player_ev - Fraction(3, 2)
    assert solve_river_equilibrium(game).certificate.gap == 0


def test_correlated_deals_and_swapped_player_order():
    original = polar_river_spot()
    swapped = JointMass(tuple(JointEntry(e.second_index, e.first_index, e.weight) for e in original.joint_mass.entries))
    game = compile_river_spot(RiverSpot(original.board, swapped, first_player=1))
    result = solve_river_equilibrium(game)
    assert result.certificate.value == Fraction(-1, 4)
    assert result.certificate.gap == 0
    correlated = JointMass((
        JointEntry(HAND_INDEX[hand("Jc", "Ad")], HAND_INDEX[hand("9c", "Kd")], Fraction(2, 3)),
        JointEntry(HAND_INDEX[hand("4d", "5d")], HAND_INDEX[hand("8c", "Qd")], Fraction(1, 3)),
    ))
    game = compile_river_spot(RiverSpot(original.board, correlated))
    assert solve_river_equilibrium(game).certificate.gap == 0


def test_board_ties_and_odd_pot_center_exactly():
    spot = RiverSpot.from_ranges(
        hand("Tc", "Jd", "Qh", "Ks", "Ac"),
        RangeMass.uniform([hand("2c", "3d")]), RangeMass.uniform([hand("4d", "5h")]), pot=3,
    )
    game = compile_river_spot(spot)
    result = solve_river_equilibrium(game)
    assert result.certificate.value == result.certificate.gap == 0
    assert verify_certificate(certificate_to_dict(game, result.profile)).gap == 0


def test_small_game_limits_and_invalid_deals(benchmark):
    game, _ = benchmark
    with pytest.raises(ValueError, match="too many pure plans"):
        compile_river_spot(game.spot, max_plans=4)
    with pytest.raises(ValueError, match="too many deals"):
        compile_river_spot(game.spot, max_deals=1)
    with pytest.raises(TypeError, match="must be integers"):
        RiverSpot(game.spot.board, game.spot.joint_mass, pot=1.0)
    overlap = JointMass((JointEntry(HAND_INDEX[hand("2c", "Ah")], HAND_INDEX[hand("9c", "Kd")], Fraction(1)),))
    with pytest.raises(ValueError, match="overlaps the board"):
        RiverSpot(game.spot.board, overlap)


def test_committed_certificate_can_be_verified():
    path = Path(__file__).parents[1] / "examples" / "polar_river_certificate.json"
    checked = verify_certificate(json.loads(path.read_text(encoding="utf-8")))
    assert checked.value == Fraction(1, 4) and checked.gap == 0
