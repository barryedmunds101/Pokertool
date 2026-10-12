# Exact river equilibrium

Board: 2c 3c 7d 9h Js
Initial pot: 1 blind; fixed bet: 1 blind.
The full one-bet tree includes betting after a check.

Legal pure plans: 16 x 4.
Exact rational LP pivots: 6.
Centered first-player value: 1/4 blind.
Original chip gains: P0 = 3/4; P1 = 1/4.
First-player lower and upper bounds: 1/4, 1/4.
Unilateral improvement: P0 = 0; P1 = 0.
Best-response gap: 0. Exact equilibrium: True.

Values below subtract half the initial pot from each player's original chip gain.

## P0: 4d 5d at root

Actual reach mass: 1/2; counterfactual mass: 1/2.

| Action | Probability | Conditional continuation value |
| --- | ---: | ---: |
| check | 1/2 | -1/2 |
| bet | 1/2 | -1/2 |

Mixed actions have exactly equal continuation value.
Opponent hand probabilities: 9c Kd: 1.
Showdown equity against those beliefs: 0.

## P0: Jc Ad at root

Actual reach mass: 1/2; counterfactual mass: 1/2.

| Action | Probability | Conditional continuation value |
| --- | ---: | ---: |
| check | 0 | 1 |
| bet | 1 | 1 |

Bet attains the largest continuation value.
Opponent hand probabilities: 9c Kd: 1.
Showdown equity against those beliefs: 1.

## P0: 4d 5d at check / bet

Actual reach mass: 1/8; counterfactual mass: 1/4.

| Action | Probability | Conditional continuation value |
| --- | ---: | ---: |
| fold | 1 | -1/2 |
| call | 0 | -3/2 |

Fold attains the largest continuation value.
Opponent hand probabilities: 9c Kd: 1.
Showdown equity against those beliefs: 0.
A call breaks even against folding at equity 1/3.

## P0: Jc Ad at check / bet

Actual reach mass: 0; counterfactual mass: 1/4.

| Action | Probability | Conditional continuation value |
| --- | ---: | ---: |
| fold | 0 | -1/2 |
| call | 1 | 3/2 |

Your strategy never reaches this information set; values describe a counterfactual continuation.
Opponent hand probabilities: 9c Kd: 1.
Showdown equity against those beliefs: 1.
A call breaks even against folding at equity 1/3.

## P1: 9c Kd at bet

Actual reach mass: 3/4; counterfactual mass: 3/4.

| Action | Probability | Conditional continuation value |
| --- | ---: | ---: |
| fold | 1/2 | -1/2 |
| call | 1/2 | -1/2 |

Mixed actions have exactly equal continuation value.
Opponent hand probabilities: Jc Ad: 2/3; 4d 5d: 1/3.
Showdown equity against those beliefs: 1/3.
A call breaks even against folding at equity 1/3.

## P1: 9c Kd at check

Actual reach mass: 1/4; counterfactual mass: 1/4.

| Action | Probability | Conditional continuation value |
| --- | ---: | ---: |
| check | 1/2 | 1/2 |
| bet | 1/2 | 1/2 |

Mixed actions have exactly equal continuation value.
Opponent hand probabilities: Jc Ad: 0; 4d 5d: 1.
Showdown equity against those beliefs: 1.

The certificate covers this embedded board, joint deal mass and betting tree.
It checks all complete information-set deviations, including changes at several decisions.
It does not claim an equilibrium for larger ranges or additional betting actions.
