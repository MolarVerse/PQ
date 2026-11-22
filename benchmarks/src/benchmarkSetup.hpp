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

#ifndef BENCHMARKS__SRC__BENCHMARK_SETUP_HPP
#define BENCHMARKS__SRC__BENCHMARK_SETUP_HPP

#include <cstddef>

#include "atom.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "simulationBox.hpp"
#include "strongTypes.hpp"
#include "units.hpp"
#include "vector3d.hpp"

namespace benchmarkSetup
{
    inline constexpr Pos cellEdge = 3.0 * angstrom;
    inline constexpr Pos cutoff   = 4.0 * angstrom;

    inline molsys::SimulationBox makeLattice(const std::size_t cellsPerSide)
    {
        const auto boxEdge = cellEdge * static_cast<double>(cellsPerSide);

        molsys::OrthorhombicBox box{{boxEdge, boxEdge, boxEdge}};
        molsys::SimulationBox   simulationBox;
        simulationBox.setBox(box, false);

        std::size_t atomIndex = 0;
        for (std::size_t x = 0; x < cellsPerSide; ++x)
        {
            for (std::size_t y = 0; y < cellsPerSide; ++y)
            {
                for (std::size_t z = 0; z < cellsPerSide; ++z)
                {
                    auto        atom     = std::make_shared<molsys::Atom>();
                    const Pos3D position = {
                        (-boxEdge / 2.0) +
                            ((static_cast<double>(x) + 0.5) * cellEdge),
                        (-boxEdge / 2.0) +
                            ((static_cast<double>(y) + 0.5) * cellEdge),
                        (-boxEdge / 2.0) +
                            ((static_cast<double>(z) + 0.5) * cellEdge),
                    };
                    atom->setPosition(position);
                    atom->setPositionOld(position);
                    atom->setVelocity(
                        linalg::Vec3D{
                            0.001 * static_cast<double>(atomIndex + 1),
                            -0.015,
                            0.02,
                        } *
                        angstrom / s
                    );
                    atom->setForce(
                        linalg::Vec3D{0.1, -0.2, 0.05} * kcal_per_mol / angstrom
                    );
                    atom->setMass(12.0 * g_per_mol);
                    atom->setAtomType(AtomType{0});
                    atom->setInternalGlobalVDWType(VdwType{0});
                    const auto charge = atomIndex++ % 2 == 0 ? 0.4 : -0.4;
                    atom->setPartialCharge(charge * elementary_charge);
                    atom->setShiftForce(
                        linalg::Vec3D{0.0, 0.0, 0.0} * kcal_per_mol
                    );

                    molsys::Molecule molecule;
                    molecule.setMoltype(MolType{1});
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

    inline void resetForces(molsys::SimulationBox& simulationBox)
    {
        simulationBox.resetForces();
        for (const auto& atom : simulationBox.getAtoms())
            atom->setShiftForce(linalg::Vec3D{0.0, 0.0, 0.0} * kcal_per_mol);
    }
}   // namespace benchmarkSetup

#endif   // BENCHMARKS__SRC__BENCHMARK_SETUP_HPP
