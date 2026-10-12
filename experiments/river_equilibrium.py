"""Run or independently verify the small exact river equilibrium benchmark.

From the repository root:
    python -m experiments.river_equilibrium
    python -m experiments.river_equilibrium --certificate certificate.json
    python -m experiments.river_equilibrium --verify certificate.json
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from cards import hand_names, make_hand
from mass import RangeMass
from river_equilibrium import (
    RiverSpot, certificate_to_dict, compile_river_spot, explain_profile,
    solve_river_equilibrium, verify_certificate,
)
from river_game import CALL


def polar_river_spot() -> RiverSpot:
    """Actual Hold'em cards: a value hand and a bluff versus one bluff catcher."""
    return RiverSpot.from_ranges(
        make_hand(["2c", "7d", "9h", "Js", "3c"]),
        RangeMass.uniform([make_hand(["Jc", "Ad"]), make_hand(["4d", "5d"])]),
        RangeMass.uniform([make_hand(["9c", "Kd"])]),
        pot=1, bet_size=1,
    )


def decision_report(game, equilibrium) -> str:
    certificate = equilibrium.certificate
    half_pot = Fraction(game.spot.pot, 2)
    lines = [
        "# Exact river equilibrium", "",
        f"Board: {' '.join(hand_names(game.spot.board))}",
        f"Initial pot: {game.spot.pot} blind; fixed bet: {game.spot.bet_size} blind.",
        "The full one-bet tree includes betting after a check.", "",
        f"Legal pure plans: {len(game.plans[0])} x {len(game.plans[1])}.",
        f"Exact rational LP pivots: {equilibrium.matrix_solution.pivots}.",
        f"Centered first-player value: {certificate.value} blind.",
        f"Original chip gains: P0 = {certificate.value + half_pot}; P1 = {half_pot - certificate.value}.",
        f"First-player lower and upper bounds: {certificate.lower_bound}, {certificate.upper_bound}.",
        f"Unilateral improvement: P0 = {certificate.first_gain}; P1 = {certificate.second_gain}.",
        f"Best-response gap: {certificate.gap}. Exact equilibrium: {certificate.exact_equilibrium}.", "",
        "Values below subtract half the initial pot from each player's original chip gain.",
    ]
    for row in explain_profile(game, equilibrium.profile):
        history = " / ".join(row.key.action_history) or "root"
        lines.extend([
            "", f"## P{row.key.player}: {' '.join(hand_names(row.key.own_hand))} at {history}", "",
            f"Actual reach mass: {row.reach}; counterfactual mass: {row.counterfactual_mass}.", "",
            "| Action | Probability | Conditional continuation value |",
            "| --- | ---: | ---: |",
        ])
        for action, probability in row.probabilities.items():
            value = str(row.action_values[action]) if row.action_values is not None else "unreached"
            lines.append(f"| {action} | {probability} | {value} |")
        lines.extend(["", row.note])
        if row.opponent_beliefs:
            beliefs = "; ".join(f"{' '.join(hand_names(hand))}: {mass}"
                                for hand, mass in row.opponent_beliefs.items())
            lines.append(f"Opponent hand probabilities: {beliefs}.")
            lines.append(f"Showdown equity against those beliefs: {row.showdown_equity}.")
            if CALL in row.probabilities:
                threshold = Fraction(game.spot.bet_size, game.spot.pot + 2 * game.spot.bet_size)
                lines.append(f"A call breaks even against folding at equity {threshold}.")
    lines.extend([
        "", "The certificate covers this embedded board, joint deal mass and betting tree.",
        "It checks all complete information-set deviations, including changes at several decisions.",
        "It does not claim an equilibrium for larger ranges or additional betting actions.", "",
    ])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--certificate", type=Path, help="save a self-contained exact JSON certificate")
    choice.add_argument("--verify", type=Path, help="verify a certificate without running the LP solver")
    args = parser.parse_args(argv)
    if args.verify is not None:
        checked = verify_certificate(json.loads(args.verify.read_text(encoding="utf-8")))
        print(f"Verified exact equilibrium: value={checked.value}, gap={checked.gap}.")
        return 0
    game = compile_river_spot(polar_river_spot())
    equilibrium = solve_river_equilibrium(game)
    print(decision_report(game, equilibrium))
    if args.certificate is not None:
        data = certificate_to_dict(game, equilibrium.profile)
        args.certificate.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Saved certificate: {args.certificate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
