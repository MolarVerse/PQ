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

#include <memory>

#include "atom.hpp"
#include "celllist.hpp"
#include "coulombPotential.hpp"
#include "coulombShiftedPotential.hpp"
#include "generalSettings.hpp"
#include "guffNonCoulomb.hpp"
#include "hybridConfigurator.hpp"
#include "hybridSettings.hpp"
#include "lennardJonesPair.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "physicalData.hpp"
#include "potentialBruteForce.hpp"
#include "potentialCellList.hpp"
#include "potentialSettings.hpp"
#include "qmmmMDEngine.hpp"
#include "simulationBox.hpp"
#include "strongTypes.hpp"
#include "virial.hpp"

class HotspotWeightingTest : public ::testing::Test
{
   public:
    static void calculateHotspotMMForces(
        engine::QMMMMDEngine &engine,
        linalg::tensor3D     &virial
    )
    {
        engine._calculateHotspotMMForces(virial);
    }
};

namespace
{
    constexpr double kBoxEdge = 15.0;
    constexpr double kCutOff  = 4.0;
    constexpr size_t kCells   = 3;
    constexpr double kSmF     = 0.3;
    constexpr double kQMM     = 0.4;    // MM charge of smoothing atom
    constexpr double kQQM     = -0.2;   // QM charge of smoothing atom
    constexpr double kQOuter  = -0.6;   // MM charge of outer atom
    constexpr double kTol     = 1.0e-10;

    // pair crosses the x boundary: dx = 12.5 -> minimum image -2.5
    const linalg::Vec3D kPosS{6.5, 0.8, 0.6};
    const linalg::Vec3D kPosM{-6.0, 0.2, -0.1};

    std::shared_ptr<pot::GuffNonCoulomb> buildGuff()
    {
        auto guff = std::make_shared<pot::GuffNonCoulomb>();
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

        const auto pair = std::make_shared<pot::LennardJonesPair>(
            kCutOff,
            LJParams{.c6 = -1.0, .c12 = 1.0}
        );

        for (size_t m1 = 1; m1 <= 2; ++m1)
        {
            for (size_t m2 = 1; m2 <= 2; ++m2)
            {
                guff->setGuffNonCoulPair(
                    {MolType{m1}, MolType{m2}, AtomType{0}, AtomType{0}},
                    pair
                );
            }
        }

        return guff;
    }

    molsys::MoleculeType buildMolType(MolType molType, double charge)
    {
        molsys::MoleculeType moleculeType;
        moleculeType.setMoltype(molType);
        moleculeType.setNumberOfAtoms(1);
        moleculeType.addExternalAtomType(ExtAtomType{molType.get()});
        moleculeType.addExternalToInternalAtomTypeElement(
            ExtAtomType{molType.get()},
            AtomType{0}
        );
        moleculeType.addPartialCharge(charge);
        moleculeType.addAtomType(AtomType{0});
        return moleculeType;
    }

    std::shared_ptr<molsys::Atom> buildAtom(
        const linalg::Vec3D &pos,
        MolType              molType,
        double               charge
    )
    {
        auto atom = std::make_shared<molsys::Atom>();
        atom->setPosition(pos);
        atom->setAtomType(AtomType{0});
        atom->setExternalAtomType(ExtAtomType{molType.get()});
        atom->setPartialCharge(charge);
        atom->setInternalGlobalVDWType(VdwType{0});
        atom->setMass(1.0);
        atom->setForceToZero();
        return atom;
    }

    void setupBox(molsys::SimulationBox &simulationBox)
    {
        molsys::OrthorhombicBox orthoBox{{kBoxEdge, kBoxEdge, kBoxEdge}};
        simulationBox.setBox(orthoBox, false);

        simulationBox.addMoleculeType(buildMolType(MolType{1}, kQMM));
        simulationBox.addMoleculeType(buildMolType(MolType{2}, kQOuter));

        auto atomS = buildAtom(kPosS, MolType{1}, kQMM);
        atomS->setQMCharge(kQQM);
        molsys::Molecule molS;
        molS.setMoltype(MolType{1});
        molS.addAtom(atomS);
        molS.setHybridZone(molsys::HybridZone::SMOOTHING);
        molS.setSmoothingFactor(kSmF);

        auto             atomM = buildAtom(kPosM, MolType{2}, kQOuter);
        molsys::Molecule molM;
        molM.setMoltype(MolType{2});
        molM.addAtom(atomM);
        molM.setHybridZone(molsys::HybridZone::OUTER);

        simulationBox.addMolecule(molS);
        simulationBox.addMolecule(molM);
        simulationBox.addAtom(atomS);
        simulationBox.addAtom(atomM);
    }

    struct PairRef
    {
        linalg::Vec3D force;   // on smoothing atom
        linalg::Vec3D shift;
        double        energy;
    };

    // reference: one unweighted kernel evaluation with the given charge tag
    template <typename Tag>
    PairRef reference(molsys::SimulationBox &simulationBox)
    {
        pot::PotentialBruteForce potential;
        potential.makeCoulombPotential(pot::CoulombShiftedPotential(kCutOff));
        potential.setNonCoulombPotential(buildGuff());

        auto atomS = buildAtom(kPosS, MolType{1}, kQMM);
        atomS->setQMCharge(kQQM);
        auto atomM = buildAtom(kPosM, MolType{2}, kQOuter);
        atomS->setShiftForce(0.0);

        const auto [coulombE, nonCoulombE] =
            potential.calculateSingleInteraction<Tag, pot::MMChargeTag>(
                *simulationBox.getBoxPtr(),
                simulationBox.getMolecule(0),
                simulationBox.getMolecule(1),
                *atomS,
                *atomM
            );

        EXPECT_NEAR(norm(atomS->getForce() + atomM->getForce()), 0.0, kTol);
        return {
            atomS->getForce(),
            atomS->getShiftForce(),
            coulombE + nonCoulombE
        };
    }

    template <typename PotentialT>
    void runHotspotCase(bool useCellList)
    {
        settings::GeneralSettings::setJobtype(JobType::QMMM_MD);
        settings::GeneralSettings::setVirialType(VirialType::ATOMIC);
        settings::HybridSettings::setUseQMCharges(true);
        settings::HybridSettings::setSmoothingMethod(SmoothingMethod::HOTSPOT);
        settings::PotentialSettings::setCoulombRadiusCutOff(kCutOff);
        pot::CoulombPotential::setCoulombRadiusCutOff(kCutOff);
        pot::CoulombPotential::setCoulombEnergyCutOff(0.0);
        pot::CoulombPotential::setCoulombForceCutOff(0.0);

        if (useCellList)
        {
            settings::GeneralSettings::activateCellList();
        }
        else
        {
            settings::GeneralSettings::deactivateCellList();
        }

        Settings             settings;
        engine::QMMMMDEngine engine{settings};
        engine.makePotential(PotentialT());
        engine.getPotential()->makeCoulombPotential(
            pot::CoulombShiftedPotential(kCutOff)
        );
        engine.getPotential()->setNonCoulombPotential(buildGuff());

        auto &simulationBox = engine.getSimulationBox();
        setupBox(simulationBox);

        if (useCellList)
        {
            auto &cellList = engine.getCellList();
            cellList.setNumberOfCells(kCells);
            cellList.resizeCells();
            cellList.setup(simulationBox);
            cellList.updateCellList(simulationBox);
        }

        const auto qmRef = reference<pot::QMChargeTag>(simulationBox);
        const auto mmRef = reference<pot::MMChargeTag>(simulationBox);

        // engine sequence of _applyHotspotSmoothing up to step 3; the QM
        // run of step 1 is skipped (no QM runner), its forces are zero
        configurator::HybridConfigurator::activateMolecules(simulationBox);
        configurator::HybridConfigurator::deactivateOuterMolecules(
            simulationBox
        );
        simulationBox.resetForces();
        engine.getPhysicalDataPtr()->resetEnergies();

        linalg::tensor3D virial{0.0};
        HotspotWeightingTest::calculateHotspotMMForces(engine, virial);

        auto      &atomS  = *simulationBox.getMolecule(0).getAtoms()[0];
        auto      &atomM  = *simulationBox.getMolecule(1).getAtoms()[0];
        const auto forceS = atomS.getForceOuter();
        const auto forceM = atomM.getForceOuter();

        // physically expected: pairwise weighted, antisymmetric
        const auto fExp = (kSmF * qmRef.force) + ((1.0 - kSmF) * mmRef.force);
        const auto shiftExp =
            (kSmF * qmRef.shift) + ((1.0 - kSmF) * mmRef.shift);
        const auto virialExp = tensorProduct(kPosS, fExp) +
                               tensorProduct(kPosM, -fExp) +
                               diagonalMatrix(shiftExp);
        const auto eExp = (kSmF * qmRef.energy) + ((1.0 - kSmF) * mmRef.energy);
        const auto eObs = engine.getPhysicalData().getCoulombEnergy() +
                          engine.getPhysicalData().getNonCoulombEnergy();

        for (size_t k = 0; k < 3; ++k)
        {
            EXPECT_NEAR(forceS[k], fExp[k], kTol) << "F_S[" << k << "]";
            EXPECT_NEAR(forceM[k], -fExp[k], kTol) << "F_M[" << k << "]";
            EXPECT_NEAR(forceS[k] + forceM[k], 0.0, kTol) << "net[" << k << "]";
            for (size_t l = 0; l < 3; ++l)
            {
                EXPECT_NEAR(virial[k][l], virialExp[k][l], kTol)
                    << "virial[" << k << "][" << l << "]";
            }
        }
        EXPECT_NEAR(eObs, eExp, kTol) << "energy";

        // pair really crosses the boundary
        EXPECT_GT(norm(qmRef.shift), 0.0);

        settings::GeneralSettings::deactivateCellList();
    }

}   // namespace

TEST_F(HotspotWeightingTest, BruteForceSmoothingOuterPair)
{
    runHotspotCase<pot::PotentialBruteForce>(false);
}

TEST_F(HotspotWeightingTest, CellListSmoothingOuterPair)
{
    runHotspotCase<pot::PotentialCellList>(true);
}
