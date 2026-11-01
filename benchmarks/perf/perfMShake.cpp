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

// Fixed-work micro-benchmark of the M-SHAKE / M-RATTLE solver for rigid
// three-site water molecules (the constraint of rigid water MD). The molecules
// start at rest in the reference geometry; every iteration one hydrogen per
// molecule is displaced, so each call has to correct a violated geometry as in
// a real step.

#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "atom.hpp"
#include "mShake.hpp"
#include "mShakeReference.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "orthorhombicBox.hpp"
#include "perfAlloc.hpp"
#include "simulationBox.hpp"
#include "strongTypes.hpp"
#include "timingsSettings.hpp"
#include "vector3d.hpp"

static constexpr std::uint64_t ITERATIONS     = 1000;
static constexpr std::size_t   N_MOLECULES    = 100;
static constexpr std::size_t   PER_SIDE       = 5;
static constexpr double        SPACING        = 8.0;
static constexpr std::size_t   MAX_ITERATIONS = 100;
static constexpr double        TOLERANCE      = 1.0e-8;

int main()
{
    settings::TimingsSettings::setTimeStep(0.5);

    // rigid water: oxygen and two hydrogens
    const auto names  = std::vector<std::string>{"O", "H", "H"};
    const auto refPos = std::vector<linalg::Vec3D>{
        {0.0, 0.0, 0.0},
        {0.585882, 0.756950, 0.0},
        {0.585882, -0.756950, 0.0}
    };

    auto moleculeType = molsys::MoleculeType();
    moleculeType.setMoltype(MolType{1});
    moleculeType.setName("water");
    moleculeType.setNumberOfAtoms(3);

    auto refAtoms = std::vector<molsys::Atom>(3);
    for (std::size_t i = 0; i < 3; ++i)
    {
        refAtoms.at(i).setName(names.at(i));
        refAtoms.at(i).setPosition(refPos.at(i));
    }

    auto mShakeReference = constraints::MShakeReference();
    mShakeReference.setMoleculeType(moleculeType);
    mShakeReference.setAtoms(refAtoms);

    auto mShake = constraints::MShake();
    mShake.addMShakeReference(mShakeReference);
    mShake.initMShake();

    auto                    box = molsys::SimulationBox();
    molsys::OrthorhombicBox orthoBox{{100.0, 100.0, 100.0}};
    box.setBox(orthoBox, false);

    for (std::size_t molecule = 0; molecule < N_MOLECULES; ++molecule)
    {
        const linalg::Vec3D origin{
            SPACING * static_cast<double>(molecule % PER_SIDE),
            SPACING * static_cast<double>((molecule / PER_SIDE) % PER_SIDE),
            SPACING * static_cast<double>(molecule / (PER_SIDE * PER_SIDE))
        };

        auto water = molsys::Molecule();
        water.setMoltype(MolType{1});

        for (std::size_t i = 0; i < 3; ++i)
        {
            auto atom = std::make_shared<molsys::Atom>();
            atom->setName(names.at(i));
            atom->initMass();
            atom->setPosition(origin + refPos.at(i));
            atom->setPositionOld(origin + refPos.at(i));
            atom->setVelocity({0.01, -0.02 * static_cast<double>(i), 0.005});
            water.addAtom(atom);
        }

        box.addMolecule(water);
    }

    perfAllocBegin();
    CALLGRIND_ZERO_STATS;

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        // without this the geometry is satisfied after the first call
        for (auto &molecule : box.getMolecules())
        {
            auto &atom  = molecule.getAtom(AtomIndex{1});
            auto  pos   = atom.getPosition();
            pos[0]     += (i % 2 == 0) ? 0.02 : -0.02;
            atom.setPosition(pos);
        }

        mShake.applyMShake(box, MAX_ITERATIONS, TOLERANCE);
        mShake.applyMRattle(box);
    }

    // read state so the loop cannot be optimized away: the O-H bond length of
    // the first molecule (the reference value is restored by M-SHAKE)
    const auto &first = box.getMolecules().front();
    const auto  bond  = norm(
        first.getAtomPosition(AtomIndex{1}) -
        first.getAtomPosition(AtomIndex{0})
    );
    std::cout << std::format("{:.6f}\n", bond);
    return 0;
}
