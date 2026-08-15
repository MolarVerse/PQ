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

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "atom.hpp"
#include "celllist.hpp"
#include "coulombPotential.hpp"
#include "coulombShiftedPotential.hpp"
#include "guffNonCoulomb.hpp"
#include "lennardJonesPair.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "physicalData.hpp"
#include "potentialBruteForce.hpp"
#include "potentialCellList.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"

using linearAlgebra::Vec3D;
using molsys::Atom;
using molsys::CellList;
using molsys::Molecule;
using molsys::MoleculeType;
using molsys::SimulationBox;
using physicalData::PhysicalData;
using potential::CoulombPotential;
using potential::CoulombShiftedPotential;
using potential::GuffNonCoulomb;
using potential::LennardJonesPair;
using potential::NonCoulombPair;
using potential::PotentialBruteForce;
using potential::PotentialCellList;
using settings::PotentialSettings;

namespace
{

    // Box / cell layout is chosen so that the periodic neighbour offsets do not
    // alias the same physical cell: with nNeighbour = ceil(cutoff / cellSize)
    // we must have kCellsPerSide >= 2 * nNeighbour + 1. Here cellSize = 5 and
    // cutoff = 4 yield nNeighbour = 1, so kCellsPerSide = 3 is sufficient and
    // the half-neighbour list in CellList visits each cell pair exactly once.
    constexpr double kBoxEdge        = 15.0;
    constexpr double kCoulombCutOff  = 4.0;
    constexpr size_t kCellsPerSide   = 3;
    constexpr double kForceTolerance = 1.0e-10;

    struct Placement
    {
        size_t molType;
        Vec3D  position;
    };

    /*
     * Build a SimulationBox with two single-atom molecule types and a handful
     * of molecules placed so the workload exercises both code paths in
     * PotentialCellList: pairs inside the same cell and pairs across
     * neighbouring cells, with some pairs above and below the cutoff.
     *
     * Each call constructs independent Atom shared_ptrs, so two boxes built
     * from the same placements are completely decoupled and can be
     * force-evaluated with the two potentials in parallel.
     */
    SimulationBox buildSimulationBox(const std::vector<Placement> &placements)
    {
        SimulationBox simBox;
        simBox.setBoxDimensions({kBoxEdge, kBoxEdge, kBoxEdge});

        auto buildMoleculeType = [](const size_t molType, const double charge)
        {
            MoleculeType mt;
            mt.setMoltype(molType);
            mt.setNumberOfAtoms(1);
            mt.addExternalAtomType(molType);
            mt.addExternalToInternalAtomTypeElement(molType, 0);
            mt.addPartialCharge(charge);
            mt.addAtomType(0);
            return mt;
        };

        simBox.addMoleculeType(buildMoleculeType(1, 0.5));
        simBox.addMoleculeType(buildMoleculeType(2, -0.3));

        for (const auto &p : placements)
        {
            auto atom = std::make_shared<Atom>();
            atom->setPosition(p.position);
            atom->setAtomType(0);
            atom->setExternalAtomType(p.molType);
            atom->setPartialCharge(p.molType == 1 ? 0.5 : -0.3);
            atom->setInternalGlobalVDWType(0);
            atom->setForceToZero();

            Molecule molecule;
            molecule.setMoltype(p.molType);
            molecule.setNumberOfAtoms(1);
            molecule.addAtom(atom);

            simBox.addMolecule(molecule);
        }

        return simBox;
    }

    std::shared_ptr<GuffNonCoulomb> buildGuffNonCoulomb()
    {
        auto guff = std::make_shared<GuffNonCoulomb>();
        guff->resizeGuff(2);
        for (size_t m1 = 0; m1 < 2; ++m1)
        {
            guff->resizeGuff(m1, 2);
            for (size_t m2 = 0; m2 < 2; ++m2)
            {
                guff->resizeGuff(m1, m2, 1);
                guff->resizeGuff(m1, m2, 0, 1);
            }
        }

        const auto pair = std::make_shared<LennardJonesPair>(
            kCoulombCutOff,
            /*c6=*/-1.0,
            /*c12=*/1.0
        );

        for (size_t m1 = 1; m1 <= 2; ++m1)
            for (size_t m2 = 1; m2 <= 2; ++m2)
                guff->setGuffNonCoulPair({m1, m2, 0, 0}, pair);

        return guff;
    }

}   // namespace

TEST(PotentialEquivalence, BruteForceMatchesCellList)
{
    PotentialSettings::setCoulombRadiusCutOff(kCoulombCutOff);
    CoulombPotential::setCoulombRadiusCutOff(kCoulombCutOff);
    CoulombPotential::setCoulombEnergyCutOff(0.0);
    CoulombPotential::setCoulombForceCutOff(0.0);

    // Cell layout: box 15 Å, 3 cells per side, cellSize 5 Å, cutoff 4 Å.
    // Cells per axis: [-7.5,-2.5], [-2.5,+2.5], [+2.5,+7.5]. Placements are
    // spread so both paths see at least one in-cutoff pair inside the same
    // cell, one in-cutoff pair across neighbouring cells (including across a
    // periodic boundary), and an out-of-cutoff pair both paths must skip.
    const std::vector<Placement> placements = {
        {1, {-5.0, -5.0, -5.0}},
        {2, {-3.0, -4.0, -3.5}},
        {1, {1.0, 1.5, 2.0}},
        {2, {2.0, 2.5, 3.5}},
        {1, {-1.0, 3.0, 1.0}},
        {2, {7.0, -7.0, 6.0}},
    };

    auto simBoxBF = buildSimulationBox(placements);
    auto simBoxCL = buildSimulationBox(placements);

    PhysicalData physicalDataBF;
    PhysicalData physicalDataCL;

    PotentialBruteForce bf;
    bf.makeCoulombPotential(CoulombShiftedPotential(kCoulombCutOff));
    bf.setNonCoulombPotential(buildGuffNonCoulomb());

    PotentialCellList cl;
    cl.makeCoulombPotential(CoulombShiftedPotential(kCoulombCutOff));
    cl.setNonCoulombPotential(buildGuffNonCoulomb());

    // Brute force ignores its CellList argument; pass a default-constructed
    // one purely to satisfy the signature.
    CellList dummyCellList;
    bf.calculateForces(simBoxBF, physicalDataBF, dummyCellList);

    CellList cellList;
    cellList.setNumberOfCells(kCellsPerSide);
    cellList.resizeCells();
    cellList.setup(simBoxCL);
    cellList.activate();
    cellList.updateCellList(simBoxCL);
    cl.calculateForces(simBoxCL, physicalDataCL, cellList);

    EXPECT_NEAR(
        physicalDataBF.getCoulombEnergy(),
        physicalDataCL.getCoulombEnergy(),
        kForceTolerance
    );
    EXPECT_NEAR(
        physicalDataBF.getNonCoulombEnergy(),
        physicalDataCL.getNonCoulombEnergy(),
        kForceTolerance
    );

    ASSERT_EQ(simBoxBF.getNumberOfMolecules(), simBoxCL.getNumberOfMolecules());
    for (size_t i = 0; i < simBoxBF.getNumberOfMolecules(); ++i)
    {
        const auto &molBF = simBoxBF.getMolecule(i);
        const auto &molCL = simBoxCL.getMolecule(i);

        ASSERT_EQ(molBF.getNumberOfAtoms(), molCL.getNumberOfAtoms());

        for (size_t a = 0; a < molBF.getNumberOfAtoms(); ++a)
        {
            const auto fBF = molBF.getAtomForce(a);
            const auto fCL = molCL.getAtomForce(a);

            EXPECT_NEAR(fBF[0], fCL[0], kForceTolerance)
                << "force x mismatch on molecule " << i << " atom " << a;
            EXPECT_NEAR(fBF[1], fCL[1], kForceTolerance)
                << "force y mismatch on molecule " << i << " atom " << a;
            EXPECT_NEAR(fBF[2], fCL[2], kForceTolerance)
                << "force z mismatch on molecule " << i << " atom " << a;
        }
    }
}
