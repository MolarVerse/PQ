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

#ifndef PQ_GOOGLE_BENCHMARK_SETUP_HPP
#define PQ_GOOGLE_BENCHMARK_SETUP_HPP

#include <cstddef>

#include "atom.hpp"
#include "molecule.hpp"
#include "simulationBox.hpp"
#include "strongTypes.hpp"
#include "vector3d.hpp"

namespace benchmarkSetup
{
    inline constexpr double cellEdge = 3.0;
    inline constexpr double cutOff   = 4.0;

    inline simulationBox::SimulationBox makeLattice(
        const std::size_t cellsPerSide
    )
    {
        const double boxEdge = cellEdge * static_cast<double>(cellsPerSide);

        simulationBox::SimulationBox simulationBox;
        simulationBox.setBoxDimensions({boxEdge, boxEdge, boxEdge});

        std::size_t atomIndex = 0;
        for (std::size_t x = 0; x < cellsPerSide; ++x)
        {
            for (std::size_t y = 0; y < cellsPerSide; ++y)
            {
                for (std::size_t z = 0; z < cellsPerSide; ++z)
                {
                    auto atom = std::make_shared<simulationBox::Atom>();
                    const linearAlgebra::Vec3D position{
                        -boxEdge / 2.0 +
                            (static_cast<double>(x) + 0.5) * cellEdge,
                        -boxEdge / 2.0 +
                            (static_cast<double>(y) + 0.5) * cellEdge,
                        -boxEdge / 2.0 +
                            (static_cast<double>(z) + 0.5) * cellEdge,
                    };
                    atom->setPosition(position);
                    atom->setPositionOld(position);
                    atom->setVelocity({
                        0.001 * static_cast<double>(atomIndex + 1),
                        -0.015,
                        0.02,
                    });
                    atom->setForce({0.1, -0.2, 0.05});
                    atom->setMass(12.0);
                    atom->setAtomType(0);
                    atom->setInternalGlobalVDWType(VdwType{0});
                    atom->setPartialCharge(atomIndex++ % 2 == 0 ? 0.4 : -0.4);
                    atom->setShiftForce({0.0, 0.0, 0.0});

                    simulationBox::Molecule molecule;
                    molecule.setMoltype(1);
                    molecule.setNumberOfAtoms(1);
                    molecule.setMolMass(12.0);
                    molecule.addAtom(atom);

                    simulationBox.addAtom(atom);
                    simulationBox.addMolecule(molecule);
                }
            }
        }

        simulationBox.calculateTotalMass();
        simulationBox.calculateDegreesOfFreedom();
        simulationBox.calculateCenterOfMass();

        return simulationBox;
    }

    inline void resetForces(simulationBox::SimulationBox& simulationBox)
    {
        simulationBox.resetForces();
        for (const auto& atom : simulationBox.getAtoms())
            atom->setShiftForce({0.0, 0.0, 0.0});
    }
}   // namespace benchmarkSetup

#endif   // PQ_GOOGLE_BENCHMARK_SETUP_HPP
