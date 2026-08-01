/*****************************************************************************
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
******************************************************************************/

// Shared construction helpers for the performance benchmarks, so each
// benchmark does not re-implement the molecule/box/potential setup. This code
// runs before CALLGRIND_ZERO_STATS in every benchmark, so it is excluded from
// the measured instruction count.

#ifndef PQ_BENCH_SETUP_HPP
#define PQ_BENCH_SETUP_HPP

#include <cstddef>
#include <memory>

#include "atom.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "lennardJonesPair.hpp"
#include "matrix.hpp"
#include "molecule.hpp"
#include "simulationBox.hpp"

namespace potential
{
    class NonCoulombPair;   // forward declaration
}

namespace benchSetup
{
    // A molecule of nAtoms on a compact lattice with mass / velocity / force /
    // shift-force / charge / atom-type / vdW-type set. Atom types alternate
    // 0/1 and charges +/-0.4.
    inline simulationBox::Molecule makeMolecule(
        const std::size_t nAtoms,
        const double      origin = 0.0
    )
    {
        auto molecule = simulationBox::Molecule();
        molecule.setMoltype(1);
        molecule.setNumberOfAtoms(nAtoms);

        double molMass = 0.0;
        for (std::size_t i = 0; i < nAtoms; ++i)
        {
            auto atom = std::make_shared<simulationBox::Atom>();

            const double d = static_cast<double>(i);
            // Quadratic y-term keeps atoms non-collinear so the bend-force
            // and dihedral kernels exercise their hot path (sin(alpha) != 0).
            const linearAlgebra::Vec3D pos{
                origin + 1.0 + 0.7 * d,
                0.4 * d + 0.1 * d * d,
                0.25 * d
            };
            atom->setPosition(pos);
            atom->setPositionOld(
                pos
            );   // at-rest start (stable for constraints)
            atom->setVelocity({0.01 * (d + 1.0), -0.015, 0.02});
            atom->setForce({0.1, -0.2, 0.05});
            atom->setShiftForce({0.0, 0.0, 0.0});
            atom->setMass(12.0);
            atom->setAtomType(i % 2);
            atom->setInternalGlobalVDWType(i % 2);
            atom->setPartialCharge((i % 2 == 0) ? 0.4 : -0.4);

            molecule.addAtom(atom);
            molMass += 12.0;
        }
        molecule.setMolMass(molMass);

        return molecule;
    }

    // A ForceFieldNonCoulomb with a Lennard-Jones pair for the 0/1 vdW types.
    inline potential::ForceFieldNonCoulomb makeNonCoulomb()
    {
        auto nonCoulomb = potential::ForceFieldNonCoulomb();
        nonCoulomb.setNonCoulombPairsMatrix(
            linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(
                2,
                2
            )
        );

        auto pair = potential::LennardJonesPair(
            std::size_t(0),
            std::size_t(1),
            12.0,
            2.0,
            3.0
        );
        nonCoulomb.setNonCoulombPairsMatrix(0, 1, pair);
        nonCoulomb.setNonCoulombPairsMatrix(1, 0, pair);

        return nonCoulomb;
    }

    // A SimulationBox populated with nMolecules of nAtomsPerMol. Both the flat
    // atom list (used by integrator/kinetics) and the molecule list (used by
    // center-of-mass/virial) are filled, and the box totals are computed.
    inline simulationBox::SimulationBox makePopulatedBox(
        const std::size_t nMolecules,
        const std::size_t nAtomsPerMol
    )
    {
        auto box = simulationBox::SimulationBox();
        box.setBoxDimensions({30.0, 30.0, 30.0});

        for (std::size_t m = 0; m < nMolecules; ++m)
        {
            auto molecule =
                makeMolecule(nAtomsPerMol, 3.0 * static_cast<double>(m));

            for (std::size_t i = 0; i < nAtomsPerMol; ++i)
                box.addAtom(molecule.getAtoms()[i]
                );   // share the atom pointers

            box.addMolecule(molecule);
        }

        box.calculateTotalMass();
        box.calculateDegreesOfFreedom();
        box.calculateCenterOfMass();

        return box;
    }
}   // namespace benchSetup

#endif   // PQ_BENCH_SETUP_HPP
