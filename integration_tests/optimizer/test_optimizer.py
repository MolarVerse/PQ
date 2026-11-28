"""
*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
*****************************************************************************

Geometry optimisation (`mm-opt`) with real steps. The system is the tilted O-H
bond of the `hessian_bond` data (harmonic, E = k / 2 (r - r0)^2, k = 500
kcal/mol/Angstrom^2, r0 = 1 Angstrom, stretched to 1.65 Angstrom at the start),
for which the optimisers can be written down in a few lines. Every test
simulates the optimiser in Python and compares the geometry of every written
frame and the numbers of the `.opt` file with it:

* steepest descent moves every atom by `learning rate * force`;
* Adam as documented in the code (beta 0.9 and 0.999, epsilon 1e-8, the first
  update has step 1);
* the learning rate strategies: constant; exponential decay,
  `lr0 * exp(-decay * step / nstep)`; constant decay, `lr - decay` after every
  `update-freq` steps, the n-th update of the learning rate has step n; and the
  minimum learning rate.

The runs end by convergence (the RMS force falls below a threshold) at an epoch
that the model predicts. The geometry of the converged step itself is not in
the written frames, so only the frames that exist are compared.
"""

import math
import pathlib
import re

import pytest

from conftest import execute_pq

K = 500.0
R0 = 1.0
START = ([-0.3, -0.4, -0.5], [0.5, 0.4, 0.7])
LR0 = 5e-4  # steepest descent: 1 - 2 k lr = 0.5 per step
NSTEP = 20
RMS_FORCE_THRESHOLD = 5.0
POSITION_TOLERANCE = 1e-7  # the frames print 8 decimals
OPT_TOLERANCE = 1e-6  # relative; the .opt file prints 9 significant digits


def bond_forces(positions):
    """Force on both atoms and the energy of the harmonic bond."""
    delta = [b - a for a, b in zip(*positions)]
    distance = math.sqrt(sum(value * value for value in delta))
    magnitude = K * (distance - R0)
    unit = [value / distance for value in delta]

    return ([magnitude * value for value in unit],
            [-magnitude * value for value in unit],
            0.5 * K * (distance - R0) ** 2)


def steepest_descent(learning_rates, updates):
    """The positions after 0, 1, ... updates; learning_rates(n) is the rate
    of the n-th update (n from 1)."""
    positions = [list(START[0]), list(START[1])]
    trajectory = [[list(p) for p in positions]]
    for number in range(1, updates + 1):
        forces = bond_forces(positions)[:2]
        positions = [[x + learning_rates(number) * f for x, f in zip(p, force)]
                     for p, force in zip(positions, forces)]
        trajectory.append([list(p) for p in positions])
    return trajectory


def adam(learning_rate, updates, beta1=0.9, beta2=0.999):
    positions = [list(START[0]), list(START[1])]
    first = [[0.0] * 3, [0.0] * 3]
    second = [[0.0] * 3, [0.0] * 3]
    trajectory = [[list(p) for p in positions]]
    for step in range(1, updates + 1):
        forces = bond_forces(positions)[:2]
        for atom in range(2):
            for axis in range(3):
                force = forces[atom][axis]
                first[atom][axis] = (beta1 * first[atom][axis]
                                     - (1 - beta1) * force)
                second[atom][axis] = (beta2 * second[atom][axis]
                                      + (1 - beta2) * force * force)
                mom1 = first[atom][axis] / (1 - beta1 ** step)
                mom2 = second[atom][axis] / (1 - beta2 ** step) + 1e-8
                positions[atom][axis] -= learning_rate * mom1 / math.sqrt(mom2)
        trajectory.append([list(p) for p in positions])
    return trajectory


def rms_force(positions):
    """The RMS over the atoms of the length of the force on an atom."""
    forces = bond_forces(positions)[:2]
    return math.sqrt(sum(sum(v * v for v in f) for f in forces) / 2)


def converged_update(trajectory):
    """The first update after which the RMS force is below the threshold."""
    for number, positions in enumerate(trajectory[1:], start=1):
        if rms_force(positions) < RMS_FORCE_THRESHOLD:
            return number
    pytest.fail("the model does not converge")


def run_opt(keywords, prefix="opt"):
    """Run mm-opt with the keywords (a dict that overrides or extends the
    template); returns the stdout, the frames of the xyz file and the rows of
    the .opt file. The run has to end normally (converged)."""
    text = pathlib.Path("opt.in").read_text()
    for key, value in keywords.items():
        if re.search(rf"^{re.escape(key)} = .*;", text, flags=re.MULTILINE):
            text = re.sub(rf"^{re.escape(key)} = .*;", f"{key} = {value};",
                          text, flags=re.MULTILINE)
        else:
            text += f"{key} = {value};\n"
    pathlib.Path("run.in").write_text(text)

    stdout, stderr = execute_pq("run.in")

    assert stderr == b""

    return stdout, read_frames(f"{prefix}.xyz"), read_opt(f"{prefix}.opt")


def read_frames(path):
    lines = pathlib.Path(path).read_text().split("\n")
    frames = []
    index = 0
    while index < len(lines) and lines[index].strip():
        count = int(lines[index].split()[0])
        frames.append([[float(v) for v in line.split()[1:4]]
                       for line in lines[index + 2:index + 2 + count]])
        index += count + 2
    return frames


def read_opt(path):
    return [[float(v) for v in line.split()[:5]]
            for line in pathlib.Path(path).read_text().splitlines()
            if line.strip()]


@pytest.fixture
def opt_input(test_with_data_dir):
    """The input template, written next to the data of hessian_bond."""
    pathlib.Path("opt.in").write_text("""jobtype = mm-opt;
force-field = bonded;
rcoulomb = 9.0;
start_file = start.rst;
parameter_file = bond.param;
topology_file = topology.top;
file_prefix = opt;
nstep = %d;
output_freq = 1;
optimizer = steepest-descent;
learning-rate-strategy = constant;
initial-learning-rate = %g;
energy-conv-strategy = absolute;
abs-energy-conv = 1.0e10;
rel-energy-conv = 1.0e10;
max-force-conv = 1.0e10;
rms-force-conv = %g;
""" % (NSTEP, LR0, RMS_FORCE_THRESHOLD))


def assert_follows(trajectory, converged, frames, opt_rows, stdout):
    """The written frames and .opt rows equal the model; the run converged at
    the epoch of the model."""
    assert b"Optimizer converged after" in stdout
    epochs = int(re.search(rb"converged after (\d+) epochs", stdout).group(1))
    # the epoch counts from 1 and includes the evaluation of the start
    assert epochs - 1 == converged, (
        f"converged after {epochs} epochs, the model after {converged} updates")

    # frame 0 is the start, then one per update, the converged one included
    assert len(frames) - 1 == converged
    for number, frame in enumerate(frames):
        for atom in range(2):
            for axis in range(3):
                assert frame[atom][axis] == pytest.approx(
                    trajectory[number][atom][axis], abs=POSITION_TOLERANCE), (
                    f"frame {number}, atom {atom}, axis {axis}")

    assert len(opt_rows) == len(frames)
    energies = [bond_forces(positions)[2] for positions in trajectory]
    for number, row in enumerate(opt_rows):
        if number == 0:
            assert row[1:] == [0.0, 0.0, 0.0, 0.0]
            continue
        forces = bond_forces(trajectory[number])[:2]
        change = abs(energies[number] - energies[number - 1])
        expected = [
            change,
            change / abs(energies[number - 1]),
            max(abs(v) for force in forces for v in force),
            rms_force(trajectory[number]),
        ]
        assert row[1:] == pytest.approx(expected, rel=OPT_TOLERANCE), (
            f"row {number} of the .opt file")


@pytest.mark.parametrize("example_dir", ["hessian_bond/"], indirect=False)
def test_steepest_descent_with_a_constant_learning_rate(opt_input):
    trajectory = steepest_descent(lambda number: LR0, NSTEP)
    converged = converged_update(trajectory)

    stdout, frames, opt_rows = run_opt({})

    # the distance error halves with every update
    for number in (1, 2, 3):
        assert math.dist(*trajectory[number]) - R0 == pytest.approx(
            (math.dist(*START) - R0) * 0.5 ** number, rel=1e-12)
    assert_follows(trajectory, converged, frames, opt_rows, stdout)


@pytest.mark.parametrize("example_dir", ["hessian_bond/"], indirect=False)
def test_steepest_descent_with_an_exponential_decay(opt_input):
    decay = 2.0
    trajectory = steepest_descent(
        lambda number: LR0 if number == 1 else
        LR0 * math.exp(-decay * (number - 1) / NSTEP), NSTEP)
    converged = converged_update(trajectory)

    stdout, frames, opt_rows = run_opt({
        "learning-rate-strategy": "exponential-decay",
        "learning-rate-decay": decay})

    assert_follows(trajectory, converged, frames, opt_rows, stdout)


@pytest.mark.parametrize("example_dir", ["hessian_bond/"], indirect=False)
def test_the_exponential_decay_defaults_to_one(opt_input):
    trajectory = steepest_descent(
        lambda number: LR0 if number == 1 else
        LR0 * math.exp(-1.0 * (number - 1) / NSTEP), NSTEP)
    converged = converged_update(trajectory)

    stdout, frames, opt_rows = run_opt({
        "learning-rate-strategy": "exponential-decay"})

    assert_follows(trajectory, converged, frames, opt_rows, stdout)


@pytest.mark.parametrize("example_dir", ["hessian_bond/"], indirect=False)
@pytest.mark.parametrize("frequency", [1, 2])
def test_steepest_descent_with_a_constant_decay(opt_input, frequency):
    decay = 3e-5
    rates = [LR0]
    rate = LR0
    for step in range(1, NSTEP):
        if step % frequency == 0:
            rate -= decay
        rates.append(rate)
    trajectory = steepest_descent(lambda number: rates[number - 1], NSTEP)
    converged = converged_update(trajectory)

    stdout, frames, opt_rows = run_opt({
        "learning-rate-strategy": "constant-decay",
        "learning-rate-decay": decay,
        "learning-rate-update-freq": frequency})

    assert_follows(trajectory, converged, frames, opt_rows, stdout)


@pytest.mark.parametrize("example_dir", ["hessian_bond/"], indirect=False)
def test_the_minimum_learning_rate_stops_a_decay(opt_input):
    minimum = 2e-4
    decay = 2e-4
    rates = [LR0]
    rate = LR0
    for _ in range(1, NSTEP):
        rate = max(rate - decay, minimum)
        rates.append(rate)
    assert rates[-1] == minimum and rates[0] > minimum
    trajectory = steepest_descent(lambda number: rates[number - 1], NSTEP)
    converged = converged_update(trajectory)

    stdout, frames, opt_rows = run_opt({
        "learning-rate-strategy": "constant-decay",
        "learning-rate-decay": decay,
        "min-learning-rate": minimum})

    assert_follows(trajectory, converged, frames, opt_rows, stdout)
    assert b"less than the minimum learning rate" in (
        pathlib.Path("opt.log").read_bytes() + stdout)


@pytest.mark.parametrize("example_dir", ["hessian_bond/"], indirect=False)
def test_adam(opt_input):
    learning_rate = 0.01  # larger rates oscillate through the minimum
    steps = 40
    trajectory = adam(learning_rate, steps)
    converged = converged_update(trajectory)
    assert converged > 20, "a real number of optimisation steps"

    stdout, frames, opt_rows = run_opt({
        "optimizer": "adam", "initial-learning-rate": learning_rate,
        "nstep": steps})

    # the first update moves every coordinate by the learning rate
    for atom in range(2):
        for axis in range(3):
            assert abs(trajectory[1][atom][axis]
                       - START[atom][axis]) == pytest.approx(
                learning_rate, rel=1e-6)
    assert_follows(trajectory, converged, frames, opt_rows, stdout)


@pytest.mark.parametrize("example_dir", ["hessian_bond/"], indirect=False)
def test_an_optimisation_that_does_not_converge_ends_with_a_warning(opt_input):
    """The run is not an error: PQ warns that it did not converge after nstep
    epochs and ends normally (exit code 0)."""
    text = pathlib.Path("opt.in").read_text()
    text = re.sub(r"nstep = \d+;", "nstep = 3;", text)
    text = re.sub(r"rms-force-conv = .*;", "rms-force-conv = 1.0e-9;", text)
    pathlib.Path("short.in").write_text(text)

    stdout, stderr = execute_pq("short.in")

    assert stderr == b""
    assert b"Optimizer did not converge after 3 epochs" in stdout
    assert b"Optimizer converged after" not in stdout
    assert b"Optimizer did not converge after 3 epochs" in (
        pathlib.Path("opt.log").read_bytes())
