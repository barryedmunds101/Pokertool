"""Small exact river equilibria, independent certificates, and explanations.

Reuse river_game's complete betting tree and information sets. Enumerate legal
deterministic information-set plans for a deliberately small normal-form LP;
never enumerate rational probability grids. Center existing constant-sum chip
utilities by half the initial pot. Larger games should use sequence form or an
iterative solver behind the same evaluation/certification interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Any

from cards import hand_names, make_hand
from exact_zero_sum import MatrixEquilibrium, solve_zero_sum
from hand_space import HAND_INDEX, PRIVATE_HANDS
from mass import JointEntry, JointMass, RangeMass, compatible_joint_mass
from river_game import (
    Action,
    InformationSetKey,
    RiverGameState,
    apply_action,
    information_set_key,
    legal_actions,
    new_river_game,
    strategy_utility,
    terminal_utility,
)
from showdown import TIE, WIN, compare_hands


Strategy = Mapping[InformationSetKey, Mapping[Action, Fraction]]
Profile = tuple[Strategy, Strategy]
Plan = tuple[Action, ...]
SCHEMA = "pokertool.river-equilibrium.v1"


@dataclass(frozen=True)
class RiverSpot:
    board: int
    joint_mass: JointMass
    pot: int = 1
    stacks: tuple[int, int] = (10, 10)
    bet_size: int = 1
    first_player: int = 0

    def __post_init__(self) -> None:
        integers = (self.pot, self.bet_size, self.first_player, *self.stacks)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("pot, stacks, bet size and first player must be integers")
        if not isinstance(self.joint_mass, JointMass):
            raise TypeError("joint_mass must be a JointMass")
        object.__setattr__(self, "joint_mass", self.joint_mass.normalized())
        for entry in self.joint_mass.entries:
            new_river_game(
                self.board, PRIVATE_HANDS[entry.first_index], PRIVATE_HANDS[entry.second_index],
                pot=self.pot, stacks=self.stacks, bet_size=self.bet_size,
                first_player=self.first_player,
            )

    @classmethod
    def from_ranges(
        cls, board: int, first: RangeMass, second: RangeMass, **kwargs: Any
    ) -> RiverSpot:
        return cls(board, compatible_joint_mass(board, first, second), **kwargs)


@dataclass(frozen=True)
class InformationSet:
    key: InformationSetKey
    actions: tuple[Action, ...]
    own_predecessors: tuple[tuple[InformationSetKey, Action], ...]


@dataclass(frozen=True)
class RiverGame:
    spot: RiverSpot
    roots: tuple[tuple[RiverGameState, Fraction], ...]
    information_sets: tuple[tuple[InformationSet, ...], tuple[InformationSet, ...]]
    plans: tuple[tuple[Plan, ...], tuple[Plan, ...]]
    payoff_matrix: tuple[tuple[Fraction, ...], ...]


def compile_river_spot(
    spot: RiverSpot, *, max_deals: int = 32, max_plans: int = 256,
    max_matrix_entries: int = 65_536,
) -> RiverGame:
    """Compile a tiny spot, keeping one decision per information set.

    Limits prevent accidentally expanding full Hold'em ranges into normal form.
    They are implementation limits, not strategic abstractions or approximations.
    """
    for limit in (max_deals, max_plans, max_matrix_entries):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("compilation limits must be positive integers")
    if len(spot.joint_mass.entries) > max_deals:
        raise ValueError("too many deals for the small exact reference solver")
    roots = tuple(
        (
            new_river_game(
                spot.board, PRIVATE_HANDS[e.first_index], PRIVATE_HANDS[e.second_index],
                pot=spot.pot, stacks=spot.stacks, bet_size=spot.bet_size,
                first_player=spot.first_player,
            ), e.weight,
        )
        for e in spot.joint_mass.entries
    )
    definitions: list[dict[InformationSetKey, InformationSet]] = [{}, {}]
    terminals = []

    def visit(state, chance, paths):
        if state.terminal:
            payoff = terminal_utility(state, 0) - Fraction(spot.pot, 2)
            terminals.append((chance * payoff, paths))
            return
        player = state.player_to_act
        assert player is not None
        key = information_set_key(state)
        definition = InformationSet(key, legal_actions(state), paths[player])
        previous = definitions[player].setdefault(key, definition)
        if previous != definition:
            raise ValueError("information set does not preserve actions and perfect recall")
        for action in definition.actions:
            next_paths = list(paths)
            next_paths[player] = paths[player] + ((key, action),)
            visit(apply_action(state, action), chance, tuple(next_paths))

    for root, chance in roots:
        visit(root, chance, ((), ()))
    infos = tuple(
        tuple(sorted(items.values(), key=lambda item: (
            len(item.key.action_history), item.key.action_history, item.key.own_hand,
        )))
        for items in definitions
    )
    sizes = []
    for player_infos in infos:
        size = 1
        for info in player_infos:
            size *= len(info.actions)
            if size > max_plans:
                raise ValueError("too many pure plans; use sequence form for larger spots")
        sizes.append(size)
    if sizes[0] * sizes[1] > max_matrix_entries:
        raise ValueError("payoff matrix exceeds the small reference solver limit")
    plans = tuple(tuple(product(*(info.actions for info in player_infos))) for player_infos in infos)
    positions = tuple({info.key: i for i, info in enumerate(player_infos)} for player_infos in infos)
    matrix = [[Fraction() for _ in plans[1]] for _ in plans[0]]
    for coefficient, paths in terminals:
        matching = tuple(
            [index for index, plan in enumerate(plans[player])
             if all(plan[positions[player][key]] == action for key, action in paths[player])]
            for player in (0, 1)
        )
        for i in matching[0]:
            for j in matching[1]:
                matrix[i][j] += coefficient
    return RiverGame(spot, roots, infos, plans, tuple(tuple(row) for row in matrix))


def _check_profile(game: RiverGame, profile: Profile) -> None:
    if len(profile) != 2:
        raise ValueError("profile must contain two player strategies")
    for player in (0, 1):
        expected = {info.key for info in game.information_sets[player]}
        if set(profile[player]) != expected:
            raise ValueError("strategy must specify exactly this player's information sets")
        for info in game.information_sets[player]:
            probabilities = profile[player][info.key]
            if set(probabilities) != set(info.actions):
                raise ValueError("strategy must specify exactly the legal actions")
            if any(isinstance(p, bool) or not isinstance(p, (int, Fraction)) or p < 0
                   for p in probabilities.values()):
                raise ValueError("strategy probabilities must be nonnegative exact rationals")
            if sum(probabilities.values(), Fraction()) != 1:
                raise ValueError("strategy probabilities must sum exactly to one")


def _policies(profile):
    return tuple(lambda key, actions, strategy=strategy: strategy[key] for strategy in profile)


def evaluate_profile(game: RiverGame, profile: Profile) -> Fraction:
    """Evaluate centered first-player EV through the original tree, not the LP."""
    _check_profile(game, profile)
    policies = _policies(profile)
    return sum((chance * (strategy_utility(root, policies, 0) - Fraction(game.spot.pot, 2))
                for root, chance in game.roots), Fraction())


def plan_strategy(game: RiverGame, player: int, plan: Plan) -> Strategy:
    if player not in (0, 1) or plan not in game.plans[player]:
        raise ValueError("not a legal player plan")
    return {
        info.key: {action: Fraction(action == selected) for action in info.actions}
        for info, selected in zip(game.information_sets[player], plan)
    }


def mixture_to_behavior(
    game: RiverGame, player: int, mixture: tuple[Fraction, ...]
) -> Strategy:
    """Recover conditional actions using own predecessor realization weights.

    Averaging action probabilities without these weights generally changes the
    strategy. At zero own reach, choose the first legal action deterministically.
    """
    if player not in (0, 1) or len(mixture) != len(game.plans[player]):
        raise ValueError("mixture has the wrong player or size")
    if any(isinstance(p, bool) or not isinstance(p, (int, Fraction)) or p < 0 for p in mixture):
        raise ValueError("mixture must contain nonnegative exact rationals")
    if sum(mixture, Fraction()) != 1:
        raise ValueError("mixture must sum exactly to one")
    positions = {info.key: i for i, info in enumerate(game.information_sets[player])}
    strategy = {}
    for i, info in enumerate(game.information_sets[player]):
        eligible = [(plan, weight) for plan, weight in zip(game.plans[player], mixture)
                    if weight and all(plan[positions[key]] == action
                                      for key, action in info.own_predecessors)]
        reach = sum((weight for _, weight in eligible), Fraction())
        strategy[info.key] = {
            action: (sum((weight for plan, weight in eligible if plan[i] == action), Fraction()) / reach
                     if reach else Fraction(action == info.actions[0]))
            for action in info.actions
        }
    return strategy


@dataclass(frozen=True)
class RiverCertificate:
    value: Fraction
    lower_bound: Fraction
    upper_bound: Fraction
    first_best_response: Plan
    second_best_response: Plan

    @property
    def first_gain(self) -> Fraction:
        return self.upper_bound - self.value

    @property
    def second_gain(self) -> Fraction:
        return self.value - self.lower_bound

    @property
    def gap(self) -> Fraction:
        return self.upper_bound - self.lower_bound

    @property
    def exact_equilibrium(self) -> bool:
        return self.gap == 0


def certify_profile(game: RiverGame, profile: Profile) -> RiverCertificate:
    """Independently check all deterministic information-set deviations.

    No optimizer, tableau, stored matrix or denominator grid is consulted. Since
    every mixed plan is a weighted average of pure plans, these bounds cover all
    legal deviations, including arbitrary real-valued mixing.
    """
    value = evaluate_profile(game, profile)
    first = [(evaluate_profile(game, (plan_strategy(game, 0, plan), profile[1])), plan)
             for plan in game.plans[0]]
    second = [(evaluate_profile(game, (profile[0], plan_strategy(game, 1, plan))), plan)
              for plan in game.plans[1]]
    upper, first_plan = max(first, key=lambda pair: pair[0])
    lower, second_plan = min(second, key=lambda pair: pair[0])
    return RiverCertificate(value, lower, upper, first_plan, second_plan)


@dataclass(frozen=True)
class RiverEquilibrium:
    profile: Profile
    matrix_solution: MatrixEquilibrium
    certificate: RiverCertificate


def solve_river_equilibrium(game: RiverGame) -> RiverEquilibrium:
    solution = solve_zero_sum(game.payoff_matrix)
    profile = (
        mixture_to_behavior(game, 0, solution.maximizer),
        mixture_to_behavior(game, 1, solution.minimizer),
    )
    # Choose sensible continuations behind the player's own zero-probability
    # actions. These changes preserve every opponent-deviation payoff because
    # the player's fixed predecessor sequence remains unreachable. Full pure
    # best responses also cover changing those predecessor actions.
    profile = tuple({key: dict(actions) for key, actions in strategy.items()} for strategy in profile)
    for player in (0, 1):
        for info in reversed(game.information_sets[player]):
            own_reach = Fraction(1)
            for key, action in info.own_predecessors:
                own_reach *= profile[player][key][action]
            if not own_reach:
                explanation = next(row for row in explain_profile(game, profile) if row.key == info.key)
                if explanation.action_values is not None:
                    selected = max(info.actions, key=lambda action: explanation.action_values[action])
                    profile[player][info.key] = {a: Fraction(a == selected) for a in info.actions}
    certificate = certify_profile(game, profile)
    if not certificate.exact_equilibrium or certificate.value != solution.value:
        raise RuntimeError("river result failed the independent tree certificate")
    return RiverEquilibrium(profile, solution, certificate)


@dataclass(frozen=True)
class DecisionExplanation:
    key: InformationSetKey
    reach: Fraction
    counterfactual_mass: Fraction
    probabilities: Mapping[Action, Fraction]
    action_values: Mapping[Action, Fraction] | None
    opponent_beliefs: Mapping[int, Fraction]
    showdown_equity: Fraction | None
    note: str


def explain_profile(game: RiverGame, profile: Profile) -> tuple[DecisionExplanation, ...]:
    """Aggregate nodes at information sets using chance-and-opponent reach.

    Conditional action values include future play and use centered chip utility.
    Off-path information sets are labelled; a Nash certificate alone does not
    claim sequential optimality at every unreachable continuation.
    """
    _check_profile(game, profile)
    policies = _policies(profile)
    totals = {}
    for player_infos in game.information_sets:
        for info in player_infos:
            totals[info.key] = {
                "reach": Fraction(), "mass": Fraction(), "beliefs": {},
                "values": {action: Fraction() for action in info.actions}, "equity": Fraction(),
            }

    def visit(state, chance, player_reach):
        if state.terminal:
            return
        player = state.player_to_act
        assert player is not None
        key = information_set_key(state)
        item = totals[key]
        cf = chance * player_reach[1 - player]
        item["reach"] += chance * player_reach[0] * player_reach[1]
        item["mass"] += cf
        opponent_hand = state.hands[1 - player]
        item["beliefs"][opponent_hand] = item["beliefs"].get(opponent_hand, Fraction()) + cf
        outcome = compare_hands(state.hands[player], opponent_hand, state.board)
        item["equity"] += cf * (Fraction(1) if outcome == WIN else Fraction(1, 2) if outcome == TIE else 0)
        for action in legal_actions(state):
            child = apply_action(state, action)
            item["values"][action] += cf * (
                strategy_utility(child, policies, player) - Fraction(game.spot.pot, 2)
            )
            next_reach = list(player_reach)
            next_reach[player] *= profile[player][key][action]
            visit(child, chance, tuple(next_reach))

    for root, chance in game.roots:
        visit(root, chance, (Fraction(1), Fraction(1)))
    result = []
    for player_infos in game.information_sets:
        for info in player_infos:
            item = totals[info.key]
            mass = item["mass"]
            values = {a: value / mass for a, value in item["values"].items()} if mass else None
            if not mass:
                note = "Opponent never reaches this information set; displayed actions are off path."
            elif not item["reach"]:
                note = "Your strategy never reaches this information set; values describe a counterfactual continuation."
            else:
                selected = [a for a, p in profile[info.key.player][info.key].items() if p]
                best = max(values.values())
                if any(values[a] < best for a in selected):
                    note = "A positive-probability action has a profitable local alternative."
                elif len(selected) > 1:
                    note = "Mixed actions have exactly equal continuation value."
                else:
                    note = f"{selected[0].capitalize()} attains the largest continuation value."
            result.append(DecisionExplanation(
                info.key, item["reach"], mass, profile[info.key.player][info.key], values,
                {hand: weight / mass for hand, weight in item["beliefs"].items()} if mass else {},
                item["equity"] / mass if mass else None, note,
            ))
    return tuple(result)


def certificate_to_dict(game: RiverGame, profile: Profile) -> dict[str, Any]:
    """Create a self-contained rational witness, with no solver state required."""
    checked = certify_profile(game, profile)
    return {
        "schema": SCHEMA,
        "spot": {
            "board": hand_names(game.spot.board), "pot": game.spot.pot,
            "stacks": list(game.spot.stacks), "bet_size": game.spot.bet_size,
            "first_player": game.spot.first_player,
            "deals": [
                {"first": hand_names(PRIVATE_HANDS[e.first_index]),
                 "second": hand_names(PRIVATE_HANDS[e.second_index]), "mass": str(e.weight)}
                for e in game.spot.joint_mass.entries
            ],
        },
        "strategies": [
            [{"own_hand": hand_names(info.key.own_hand),
              "history": list(info.key.action_history),
              "probabilities": {action: str(profile[player][info.key][action]) for action in info.actions}}
             for info in game.information_sets[player]]
            for player in (0, 1)
        ],
        "certificate": {
            "value": str(checked.value), "lower_bound": str(checked.lower_bound),
            "upper_bound": str(checked.upper_bound), "first_gain": str(checked.first_gain),
            "second_gain": str(checked.second_gain), "gap": str(checked.gap),
        },
    }


def _read_fraction(value: Any) -> Fraction:
    if not isinstance(value, str):
        raise ValueError("certificate fractions must be strings, not rounded JSON numbers")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise ValueError("invalid rational certificate value") from None


def verify_certificate(data: Mapping[str, Any]) -> RiverCertificate:
    """Rebuild the stated game and check every deviation without running an LP.

    Returns only after the claims match recomputed exact values and the gap is
    zero. A certificate pertains to its embedded spot, not to larger ranges or
    additional betting actions.
    """
    if data.get("schema") != SCHEMA:
        raise ValueError("unknown river certificate schema")
    spec = data["spot"]
    entries = tuple(
        JointEntry(HAND_INDEX[make_hand(e["first"])], HAND_INDEX[make_hand(e["second"])], _read_fraction(e["mass"]))
        for e in spec["deals"]
    )
    if any(e.weight <= 0 for e in entries) or sum((e.weight for e in entries), Fraction()) != 1:
        raise ValueError("certificate deal masses must be positive and sum exactly to one")
    spot = RiverSpot(
        make_hand(spec["board"]), JointMass(entries), spec["pot"], tuple(spec["stacks"]),
        spec["bet_size"], spec["first_player"],
    )
    game = compile_river_spot(spot)
    if len(data["strategies"]) != 2:
        raise ValueError("certificate requires two player strategies")
    profile = []
    for player in (0, 1):
        lookup = {(info.key.own_hand, info.key.action_history): info.key
                  for info in game.information_sets[player]}
        strategy = {}
        for item in data["strategies"][player]:
            coordinate = (make_hand(item["own_hand"]), tuple(item["history"]))
            if coordinate not in lookup or lookup[coordinate] in strategy:
                raise ValueError("duplicate or unknown information set in certificate")
            strategy[lookup[coordinate]] = {a: _read_fraction(p) for a, p in item["probabilities"].items()}
        profile.append(strategy)
    checked = certify_profile(game, tuple(profile))
    claims = data["certificate"]
    expected = {
        "value": checked.value, "lower_bound": checked.lower_bound, "upper_bound": checked.upper_bound,
        "first_gain": checked.first_gain, "second_gain": checked.second_gain, "gap": checked.gap,
    }
    if set(claims) != set(expected) or any(_read_fraction(claims[name]) != value for name, value in expected.items()):
        raise ValueError("certificate claims do not match independently recomputed bounds")
    if not checked.exact_equilibrium:
        raise ValueError("profile has a positive best-response gap")
    return checked
