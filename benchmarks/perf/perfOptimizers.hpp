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

#ifndef BENCHMARKS__PERF__PERF_OPTIMIZERS_HPP
#define BENCHMARKS__PERF__PERF_OPTIMIZERS_HPP

// Shared setup of the optimizer benchmarks: a box of atoms on a grid with
// forces that depend on the atom index, so the update steps move every atom.

#include <cstddef>
#include <memory>

#include "atom.hpp"
#include "orthorhombicBox.hpp"
#include "simulationBox.hpp"
#include "units.hpp"
#include "vector3d.hpp"

namespace benchOptimizers
{
    inline constexpr double SPACING = 2.0;

    inline std::shared_ptr<molsys::SimulationBox> makeBox(
        const std::size_t atoms,
        const std::size_t perSide
    )
    {
        auto box = std::make_shared<molsys::SimulationBox>();

        const auto              length = SPACING * static_cast<double>(perSide);
        molsys::OrthorhombicBox orthoBox{{length, length, length}};
        box->setBox(orthoBox, false);

        for (std::size_t i = 0; i < atoms; ++i)
        {
            const auto column = i % perSide;
            const auto row    = (i / perSide) % perSide;
            const auto layer  = i / (perSide * perSide);
            const auto phase  = static_cast<double>(i % 11) * 0.1;

            auto atom = std::make_shared<molsys::Atom>();
            atom->setPosition(
                linalg::Vec3D{
                    SPACING * static_cast<double>(column),
                    SPACING * static_cast<double>(row),
                    SPACING * static_cast<double>(layer)
                } *
                angstrom
            );
            atom->setForce({0.5 + phase, -0.25 * phase, 0.1 - phase});
            box->addAtom(atom);
        }

        return box;
    }

    inline Pos sumOfPositions(const molsys::SimulationBox &simulationBox)
    {
        Pos sum = 0.0 * angstrom;
        for (const auto &position : simulationBox.getPositions())
            sum += position[0] + position[1] + position[2];
        return sum;
    }
}   // namespace benchOptimizers

#endif   // BENCHMARKS__PERF__PERF_OPTIMIZERS_HPP
