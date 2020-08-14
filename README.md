# Catheter UKF

This project implements an unscented Kalman filter for our catheter tracking projects.
The filter is aware of the fixed-distance constraint between the distal and proximal coils.

It is based on the reference [Unscented Kalman Filtering on Riemannian Manifolds](https://www.researchgate.net/publication/257581786_Unscented_Kalman_Filtering_on_Riemannian_Manifolds) by
Søren Hauberg, François Lauze, and Kim Steenstrup Pedersen.

Language used in the filter is inspired by the reference (e.g., "Log" and "Exp").

## Usage

A specific example of use in code can be found in `cathy.cli.apply_ukf`.

The basic flow is:
- Create a ukf object using something like `ukf = catheter_ukf.UKF()`,
- Create initial parameters `x, P = ukf.ukf.estimate_initial_state(distal_coord, proximal_coord)`
- Filter points sequentially `x, P = ukf.filter(x, P, datapoint)`
- Translate states back into catheter coordinates `t, d, p = ukf.tip_and_coils(x)`

