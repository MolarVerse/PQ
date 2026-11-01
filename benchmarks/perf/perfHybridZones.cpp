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

// Fixed-work micro-benchmark of the QM/MM zone bookkeeping that runs every
// step of a QM/MM MD simulation: inner region center, shift into the center
// frame, zone assignment, (de)activation, smoothing factors and the shift
// back. One QM molecule sits in the middle of a cube of MM waters; every
// iteration all molecules are displaced a little so molecules keep crossing
// zone boundaries.

#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "atom.hpp"
#include "hybridConfigurator.hpp"
#include "hybridSettings.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "perfAlloc.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"

static constexpr std::uint64_t ITERATIONS = 300;
static constexpr std::size_t   PER_SIDE   = 8;
static constexpr double        SPACING    = 3.0;
static constexpr double        WOBBLE     = 0.4;

namespace
{
    auto makeWater(const linalg::Vec3D &origin) -> molsys::Molecule
    {
        static const auto names  = std::vector<std::string>{"O", "H", "H"};
        static const auto offset = std::vector<linalg::Vec3D>{
            {0.0, 0.0, 0.0},
            {0.585882, 0.756950, 0.0},
            {0.585882, -0.756950, 0.0}
        };

        auto water = molsys::Molecule();
        water.setMoltype(MolType{1});

        for (std::size_t i = 0; i < 3; ++i)
        {
            auto atom = std::make_shared<molsys::Atom>();
            atom->setName(names.at(i));
            atom->initMass();
            atom->setPosition(origin + offset.at(i));
            water.addAtom(atom);
        }

        return water;
    }

    void addWater(
        molsys::SimulationBox &simulationBox,
        const linalg::Vec3D   &origin
    )
    {
        const auto water = makeWater(origin);
        for (const auto &atom : water.getAtoms()) simulationBox.addAtom(atom);
        simulationBox.addMolecule(water);
    }
}   // namespace

int main()
{
    settings::HybridSettings::setCoreRadius(4.0);
    settings::HybridSettings::setLayerRadius(12.0);
    settings::HybridSettings::setSmoothingRegionThickness(3.0);
    settings::HybridSettings::setPointChargeThickness(6.0);

    auto                    box = molsys::SimulationBox();
    molsys::OrthorhombicBox orthoBox{{100.0, 100.0, 100.0}};
    box.setBox(orthoBox, false);

    const auto half = 0.5 * SPACING * static_cast<double>(PER_SIDE);

    // the QM molecule defines the inner region center
    addWater(box, {half, half, half});
    box.addInnerRegionCenterAtoms({0, 1, 2});

    for (std::size_t x = 0; x < PER_SIDE; ++x)
    {
        for (std::size_t y = 0; y < PER_SIDE; ++y)
        {
            for (std::size_t z = 0; z < PER_SIDE; ++z)
            {
                const linalg::Vec3D origin{
                    SPACING * (static_cast<double>(x) + 0.37),
                    SPACING * (static_cast<double>(y) + 0.41),
                    SPACING * (static_cast<double>(z) + 0.43)
                };
                addWater(box, origin);
            }
        }
    }

    auto configurator = configurator::HybridConfigurator();

    perfAllocBegin();
    CALLGRIND_ZERO_STATS;

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        // without this the zones are fixed after the first call
        const auto shift = (i % 2 == 0) ? WOBBLE : -WOBBLE;
        for (auto &atom : box.getAtoms())
        {
            auto pos  = atom->getPosition();
            pos[0]   += shift;
            atom->setPosition(pos);
        }

        configurator.calculateInnerRegionCenter(box);
        configurator.shiftAtomsToInnerRegionCenter(box);
        configurator::HybridConfigurator::assignHybridZones(box);
        configurator::HybridConfigurator::activateMolecules(box);
        configurator::HybridConfigurator::deactivateOuterMolecules(box);
        configurator::HybridConfigurator::calculateSmoothingFactors(box);
        configurator.shiftAtomsBackToInitialPositions(box);
    }

    // read state so the loop cannot be optimized away: the sum of all
    // smoothing factors of the last iteration
    auto sum = 0.0;
    for (const auto &molecule :
         box.getMoleculesInsideZone(molsys::HybridZone::SMOOTHING))
        sum += molecule.getSmoothingFactor();

    std::cout << std::format("{:.6f}\n", sum);
    return 0;
}
