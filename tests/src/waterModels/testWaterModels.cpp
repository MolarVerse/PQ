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

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "SPCIntraWater.hpp"
#include "atom.hpp"
#include "celllist.hpp"
#include "coulombPotential.hpp"
#include "coulombShiftedPotential.hpp"
#include "guffNonCoulomb.hpp"
#include "hybridSettings.hpp"
#include "interWater.hpp"
#include "lennardJonesPair.hpp"
#include "mTRIntraWater.hpp"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "potentialBruteForce.hpp"
#include "potentialCellList.hpp"
#include "potentialSettings.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "strongTypes.hpp"
#include "waterModelSettings.hpp"

using linearAlgebra::Vec3D;
using physicalData::PhysicalData;
using potential::CoulombPotential;
using potential::CoulombShiftedPotential;
using potential::GuffNonCoulomb;
using potential::LennardJonesPair;
using potential::MMChargeTag;
using potential::PotentialBruteForce;
using potential::PotentialCellList;
using potential::QMChargeTag;
using settings::HybridSettings;
using settings::JobType;
using settings::PotentialSettings;
using settings::Settings;
using settings::SmoothingMethod;
using settings::WaterModelSettings;
using simulationBox::Atom;
using simulationBox::CellList;
using simulationBox::HybridZone;
using simulationBox::Molecule;
using simulationBox::SimulationBox;
using waterModel::InterWater;
using waterModel::InterWaterState;
using waterModel::InterWaterStrategy;

namespace
{
    constexpr size_t kWaterType = 1;
    constexpr double kCutOff    = 4.0;

    struct WaterGeometry
    {
        double oh1;
        double oh2;
        double angle;
    };

    void addWater(
        SimulationBox       &simBox,
        const Vec3D         &origin,
        const WaterGeometry &geometry,
        const HybridZone     zone,
        const bool           active,
        const size_t         molType
    )
    {
        const auto oxygen = std::make_shared<Atom>();
        const auto h1     = std::make_shared<Atom>();
        const auto h2     = std::make_shared<Atom>();

        oxygen->setAtomicNumber(AtomNumber{8});
        oxygen->setPartialCharge(-0.82);
        oxygen->setQMCharge(-0.9);
        oxygen->setPosition(origin);
        oxygen->setAtomType(0);
        oxygen->setInternalGlobalVDWType(VdwType{0});
        oxygen->setForceToZero();

        h1->setAtomicNumber(AtomNumber{1});
        h1->setPartialCharge(0.41);
        h1->setQMCharge(0.45);
        h1->setPosition(origin + Vec3D{geometry.oh1, 0.0, 0.0});
        h1->setAtomType(1);
        h1->setInternalGlobalVDWType(VdwType{0});
        h1->setForceToZero();

        h2->setAtomicNumber(AtomNumber{1});
        h2->setPartialCharge(0.41);
        h2->setQMCharge(0.45);
        h2->setPosition(
            origin +
            Vec3D{
                geometry.oh2 * std::cos(geometry.angle),
                geometry.oh2 * std::sin(geometry.angle),
                0.0
            }
        );
        h2->setAtomType(1);
        h2->setInternalGlobalVDWType(VdwType{0});
        h2->setForceToZero();

        Molecule water;
        water.setMoltype(molType);
        water.setNumberOfAtoms(3);
        water.setHybridZone(zone);
        water.setSmoothingFactor(0.25);
        water.addAtom(oxygen);
        water.addAtom(h1);
        water.addAtom(h2);

        if (!active)
            water.deactivateMolecule();

        simBox.addAtom(oxygen);
        simBox.addAtom(h1);
        simBox.addAtom(h2);
        simBox.addMolecule(water);
    }

    void addWater(
        SimulationBox       &simBox,
        const Vec3D         &origin,
        const WaterGeometry &geometry,
        const HybridZone     zone
    )
    {
        addWater(simBox, origin, geometry, zone, true, kWaterType);
    }

    void addWater(
        SimulationBox       &simBox,
        const Vec3D         &origin,
        const WaterGeometry &geometry,
        const HybridZone     zone,
        bool                 active
    )
    {
        addWater(simBox, origin, geometry, zone, active, kWaterType);
    }

    SimulationBox makeIntraWaterBox(const WaterGeometry &geometry)
    {
        SimulationBox simBox;
        simBox.setBoxDimensions({20.0, 20.0, 20.0});
        simBox.setWaterType(kWaterType);
        addWater(simBox, {0.0, 0.0, 0.0}, geometry, HybridZone::SMOOTHING);
        return simBox;
    }

    template <typename Model>
    void expectIntraModelConservesForce(
        Model               &model,
        const WaterGeometry &geometry
    )
    {
        auto         simBox = makeIntraWaterBox(geometry);
        PhysicalData data;

        model.calculate(simBox, data);

        const auto totalForce = simBox.getMolecule(0).getAtomForce(0) +
                                simBox.getMolecule(0).getAtomForce(1) +
                                simBox.getMolecule(0).getAtomForce(2);

        EXPECT_NEAR(totalForce[0], 0.0, 1.0e-12);
        EXPECT_NEAR(totalForce[1], 0.0, 1.0e-12);
        EXPECT_NEAR(totalForce[2], 0.0, 1.0e-12);
        EXPECT_TRUE(std::isfinite(data.getBondEnergy()));
        EXPECT_TRUE(std::isfinite(data.getAngleEnergy()));
        EXPECT_GT(data.getBondEnergy(), 0.0);
    }

    std::shared_ptr<GuffNonCoulomb> makeNonCoulombPotential()
    {
        auto nonCoulomb = std::make_shared<GuffNonCoulomb>();
        nonCoulomb->resizeGuff(2);

        for (size_t mol1 = 0; mol1 < 2; ++mol1)
        {
            nonCoulomb->resizeGuff(mol1, 2);
            for (size_t mol2 = 0; mol2 < 2; ++mol2)
            {
                nonCoulomb->resizeGuff(mol1, mol2, 2);
                for (size_t atom1 = 0; atom1 < 2; ++atom1)
                    nonCoulomb->resizeGuff(mol1, mol2, atom1, 2);
            }
        }

        const auto pair = std::make_shared<LennardJonesPair>(
            kCutOff,
            LJParams{.c6 = -1.0, .c12 = 1.0}
        );

        for (size_t mol1 = 1; mol1 <= 2; ++mol1)
        {
            for (size_t mol2 = 1; mol2 <= 2; ++mol2)
            {
                for (size_t atom1 = 0; atom1 < 2; ++atom1)
                {
                    for (size_t atom2 = 0; atom2 < 2; ++atom2)
                    {
                        nonCoulomb->setGuffNonCoulPair(
                            {mol1, mol2, atom1, atom2},
                            pair
                        );
                    }
                }
            }
        }

        return nonCoulomb;
    }

    class ExposedInterWaterStrategy : public InterWaterStrategy
    {
       public:
        void calculate(
            const InterWaterState & /*state*/,
            SimulationBox & /*simBox*/,
            PhysicalData & /*data*/,
            const std::shared_ptr<potential::CoulombPotential> & /*coulomb*/,
            CellList & /*cellList*/
        ) final
        {
        }

        void calculateCoreToOuterForces(
            const InterWaterState & /*state*/,
            SimulationBox & /*simBox*/,
            PhysicalData & /*data*/,
            const std::shared_ptr<potential::CoulombPotential> & /*coulomb*/,
            CellList & /*cellList*/
        ) final
        {
        }

        void calculateLayerToOuterForces(
            const InterWaterState & /*state*/,
            SimulationBox & /*simBox*/,
            PhysicalData & /*data*/,
            const std::shared_ptr<potential::CoulombPotential> & /*coulomb*/,
            CellList & /*cellList*/
        ) final
        {
        }

        void calculateOuterToOuterForces(
            const InterWaterState & /*state*/,
            SimulationBox & /*simBox*/,
            PhysicalData & /*data*/,
            const std::shared_ptr<potential::CoulombPotential> & /*coulomb*/,
            CellList & /*cellList*/
        ) final
        {
        }

        void calculateHotspotSmoothingMMForces(
            const InterWaterState & /*state*/,
            SimulationBox & /*simBox*/,
            PhysicalData & /*data*/,
            const std::shared_ptr<potential::CoulombPotential> & /*coulomb*/,
            CellList & /*cellList*/
        ) final
        {
        }
    };

    SimulationBox makeHybridWaterBox()
    {
        constexpr WaterGeometry geometry{
            .oh1   = 0.96,
            .oh2   = 0.98,
            .angle = 1.82
        };

        SimulationBox simBox;
        simBox.setBoxDimensions({15.0, 15.0, 15.0});
        simBox.setWaterType(kWaterType);

        addWater(simBox, {-5.8, -5.5, -5.5}, geometry, HybridZone::CORE, false);
        addWater(
            simBox,
            {-5.2, -3.5, -5.5},
            geometry,
            HybridZone::CORE,
            false,
            2
        );
        addWater(
            simBox,
            {-4.2, -5.2, -5.2},
            geometry,
            HybridZone::LAYER,
            false
        );
        addWater(
            simBox,
            {-3.8, -3.2, -5.2},
            geometry,
            HybridZone::LAYER,
            false,
            2
        );
        addWater(simBox, {-1.8, -5.0, -5.0}, geometry, HybridZone::SMOOTHING);
        addWater(simBox, {-0.2, -4.8, -4.8}, geometry, HybridZone::SMOOTHING);
        addWater(
            simBox,
            {-1.0, -3.0, -5.0},
            geometry,
            HybridZone::SMOOTHING,
            true,
            2
        );
        addWater(simBox, {1.5, -4.6, -4.6}, geometry, HybridZone::OUTER);
        addWater(
            simBox,
            {-3.5, -3.5, -3.5},
            geometry,
            HybridZone::OUTER,
            true,
            2
        );

        return simBox;
    }

    CellList makeCellList(SimulationBox &simBox)
    {
        settings::Settings::activateCellList();

        CellList cellList;
        cellList.setNumberOfCells(3);
        cellList.resizeCells();
        cellList.setup(simBox);
        cellList.updateCellList(simBox);
        cellList.assignMoleculeHybridZoneIndices();
        cellList.assignWaterMoleculeIndices(simBox);
        return cellList;
    }

    void resetForces(SimulationBox &simBox)
    {
        for (auto &molecule : simBox.getMolecules())
            molecule.setAtomForcesToZero();
    }

}   // namespace

TEST(IntraWater, FlexibleSpcModelsProduceFiniteConservativeForces)
{
    HybridSettings::setSmoothingMethod(SmoothingMethod::HOTSPOT);

    waterModel::SPCFwIntraWater spcFw;
    expectIntraModelConservesForce(
        spcFw,
        {.oh1   = spcFw.getEqOHDistance() + 0.04,
         .oh2   = spcFw.getEqOHDistance() - 0.03,
         .angle = spcFw.getEqHOHAngle() + 0.05}
    );

    waterModel::qSPCFwIntraWater qSpcFw;
    expectIntraModelConservesForce(
        qSpcFw,
        {.oh1   = qSpcFw.getEqOHDistance() + 0.03,
         .oh2   = qSpcFw.getEqOHDistance() - 0.02,
         .angle = qSpcFw.getEqHOHAngle() - 0.04}
    );
}

TEST(IntraWater, MtrModelsProduceFiniteConservativeForces)
{
    waterModel::SPCMTRIntraWater spcMtr;
    HybridSettings::setSmoothingMethod(SmoothingMethod::HOTSPOT);
    expectIntraModelConservesForce(
        spcMtr,
        {.oh1 = 1.04, .oh2 = 0.97, .angle = 1.88}
    );
    EXPECT_DOUBLE_EQ(spcMtr.getEqOHDistance(), 1.0);
    EXPECT_DOUBLE_EQ(spcMtr.getEqHHDistance(), 1.632993162);

    waterModel::TIP3PMTRIntraWater tip3pMtr;
    HybridSettings::setSmoothingMethod(SmoothingMethod::EXACT);
    expectIntraModelConservesForce(
        tip3pMtr,
        {.oh1 = 1.00, .oh2 = 0.93, .angle = 1.82}
    );
    EXPECT_DOUBLE_EQ(tip3pMtr.getEqOHDistance(), 0.9572);
    EXPECT_DOUBLE_EQ(tip3pMtr.getEqHHDistance(), 1.5139);
}

TEST(InterWater, PairEvaluatorsApplySymmetricAndOneWayForces)
{
    PotentialSettings::setCoulombRadiusCutOff(kCutOff);
    CoulombPotential::setCoulombRadiusCutOff(kCutOff);
    CoulombPotential::setCoulombEnergyCutOff(0.0);
    CoulombPotential::setCoulombForceCutOff(0.0);

    SimulationBox simBox;
    simBox.setBoxDimensions({15.0, 15.0, 15.0});

    Atom atom1;
    atom1.setPosition({0.0, 0.0, 0.0});
    atom1.setPartialCharge(-0.8);
    atom1.setQMCharge(-0.9);
    atom1.setForceToZero();

    Atom atom2;
    atom2.setPosition({1.2, 0.1, 0.0});
    atom2.setPartialCharge(0.4);
    atom2.setQMCharge(0.45);
    atom2.setForceToZero();

    const auto coulomb = std::make_shared<CoulombShiftedPotential>(kCutOff);
    const LennardJonesPair nonCoulomb(
        kCutOff,
        LJParams{.c6 = -1.0, .c12 = 1.0}
    );
    ExposedInterWaterStrategy strategy;

    EXPECT_DOUBLE_EQ(nonCoulomb.getRadialCutOff(), kCutOff);

    double coulombEnergy    = 0.0;
    double nonCoulombEnergy = 0.0;
    strategy.calculateSingleInteraction<MMChargeTag, MMChargeTag>(
        atom1,
        atom2,
        coulomb,
        kCutOff * kCutOff,
        simBox,
        nonCoulomb,
        coulombEnergy,
        nonCoulombEnergy
    );

    EXPECT_NE(coulombEnergy, 0.0);
    EXPECT_NE(nonCoulombEnergy, 0.0);
    EXPECT_EQ(atom1.getForce(), -atom2.getForce());

    atom1.setForceToZero();
    atom2.setForceToZero();
    HybridSettings::setUseQMCharges(true);
    coulombEnergy = 0.0;
    strategy.calculateSingleCoulombInteraction<QMChargeTag, MMChargeTag>(
        atom1,
        atom2,
        coulomb,
        kCutOff * kCutOff,
        simBox,
        coulombEnergy
    );
    EXPECT_NE(coulombEnergy, 0.0);
    EXPECT_EQ(atom1.getForce(), -atom2.getForce());
    EXPECT_DOUBLE_EQ(strategy.getPartialCharge<QMChargeTag>(atom1), -0.9);

    atom1.setForceToZero();
    atom2.setForceToZero();
    coulombEnergy    = 0.0;
    nonCoulombEnergy = 0.0;
    strategy.calculateSingleInteractionOneWay<MMChargeTag, QMChargeTag>(
        atom1,
        atom2,
        coulomb,
        kCutOff * kCutOff,
        simBox,
        nonCoulomb,
        coulombEnergy,
        nonCoulombEnergy
    );
    EXPECT_NE(atom1.getForce(), Vec3D{});
    EXPECT_EQ(atom2.getForce(), Vec3D{});

    HybridSettings::setUseQMCharges(false);
    EXPECT_DOUBLE_EQ(strategy.getPartialCharge<QMChargeTag>(atom1), -0.8);
    EXPECT_DOUBLE_EQ(strategy.getPartialCharge<MMChargeTag>(atom2), 0.4);
}

TEST(InterWater, DefaultStrategyIsInert)
{
    SimulationBox simBox;
    PhysicalData  data;
    CellList      cellList;
    const auto    coulomb = std::make_shared<CoulombShiftedPotential>(kCutOff);

    InterWater interWater;
    interWater.calculate(simBox, data, coulomb, cellList);
    interWater.calculateQMMMForces(simBox, data, coulomb, cellList);
    interWater
        .calculateHotspotSmoothingMMForces(simBox, data, coulomb, cellList);

    EXPECT_DOUBLE_EQ(data.getCoulombEnergy(), 0.0);
    EXPECT_DOUBLE_EQ(data.getNonCoulombEnergy(), 0.0);
}

TEST(InterWater, NonOxygenOnlyStateInitializesEveryPair)
{
    PotentialSettings::setCoulombRadiusCutOff(kCutOff);
    PotentialSettings::setNonCoulombRadiusCutOff(kCutOff);

    auto oxygenOxygen = std::make_unique<LennardJonesPair>(
        kCutOff,
        LJParams{.c6 = -1.0, .c12 = 1.0}
    );
    auto oxygenHydrogen = std::make_unique<LennardJonesPair>(
        kCutOff,
        LJParams{.c6 = -1.0, .c12 = 1.0}
    );
    auto hydrogenHydrogen = std::make_unique<LennardJonesPair>(
        kCutOff,
        LJParams{.c6 = -1.0, .c12 = 1.0}
    );
    const auto *oxygenOxygenView     = oxygenOxygen.get();
    const auto *oxygenHydrogenView   = oxygenHydrogen.get();
    const auto *hydrogenHydrogenView = hydrogenHydrogen.get();

    InterWaterState state;
    state._oxygenOnlyNonCoulomb = false;
    state._nonCoulombPairOO     = std::move(oxygenOxygen);
    state._nonCoulombPairOH     = std::move(oxygenHydrogen);
    state._nonCoulombPairHH     = std::move(hydrogenHydrogen);

    InterWater interWater(
        std::move(state),
        std::make_unique<waterModel::InterWaterStrategyNull>()
    );
    InterWater nullPairs(
        InterWaterState{},
        std::make_unique<waterModel::InterWaterStrategyNull>()
    );
    SimulationBox simBox;
    PhysicalData  physicalData;
    CellList      cellList;
    const auto    coulomb = std::make_shared<CoulombShiftedPotential>(kCutOff);

    interWater.calculate(simBox, physicalData, coulomb, cellList);
    nullPairs.calculate(simBox, physicalData, coulomb, cellList);

    EXPECT_DOUBLE_EQ(oxygenOxygenView->getRadialCutOff(), kCutOff);
    EXPECT_DOUBLE_EQ(oxygenHydrogenView->getRadialCutOff(), kCutOff);
    EXPECT_DOUBLE_EQ(hydrogenHydrogenView->getRadialCutOff(), kCutOff);
    EXPECT_TRUE(std::isfinite(oxygenOxygenView->getEnergyCutOff()));
    EXPECT_TRUE(std::isfinite(oxygenHydrogenView->getEnergyCutOff()));
    EXPECT_TRUE(std::isfinite(hydrogenHydrogenView->getEnergyCutOff()));
    EXPECT_DOUBLE_EQ(physicalData.getCoulombEnergy(), 0.0);
    EXPECT_DOUBLE_EQ(physicalData.getNonCoulombEnergy(), 0.0);
}

TEST(InterWater, BruteForceAndCellListStrategiesExerciseHybridWaterRegions)
{
    Settings::setJobtype(JobType::QMMM_MD);
    HybridSettings::setUseQMCharges(true);
    PotentialSettings::setCoulombRadiusCutOff(kCutOff);
    PotentialSettings::setNonCoulombRadiusCutOff(kCutOff);
    CoulombPotential::setCoulombRadiusCutOff(kCutOff);
    CoulombPotential::setCoulombEnergyCutOff(0.0);
    CoulombPotential::setCoulombForceCutOff(0.0);
    WaterModelSettings::setIsInterWaterModelSet(true);

    const auto coulomb = std::make_shared<CoulombShiftedPotential>(kCutOff);

    auto         simBoxBruteForce = makeHybridWaterBox();
    CellList     unusedCellList;
    PhysicalData bruteForceData;
    InterWater   bruteForce(
        waterModel::makeInterWaterState<waterModel::SPCInterParam>(),
        std::make_unique<waterModel::InterWaterStrategyBruteForce>()
    );

    bruteForce
        .calculate(simBoxBruteForce, bruteForceData, coulomb, unusedCellList);
    resetForces(simBoxBruteForce);
    bruteForce.calculateQMMMForces(
        simBoxBruteForce,
        bruteForceData,
        coulomb,
        unusedCellList
    );
    bruteForce.calculateHotspotSmoothingMMForces(
        simBoxBruteForce,
        bruteForceData,
        coulomb,
        unusedCellList
    );

    auto         simBoxCellList = makeHybridWaterBox();
    auto         cellList       = makeCellList(simBoxCellList);
    PhysicalData cellListData;
    InterWater   cellListWater(
        waterModel::makeInterWaterState<waterModel::SPCEInterParam>(),
        std::make_unique<waterModel::InterWaterStrategyCellList>()
    );

    cellListWater.calculate(simBoxCellList, cellListData, coulomb, cellList);
    resetForces(simBoxCellList);
    cellListWater
        .calculateQMMMForces(simBoxCellList, cellListData, coulomb, cellList);
    cellListWater.calculateHotspotSmoothingMMForces(
        simBoxCellList,
        cellListData,
        coulomb,
        cellList
    );

    EXPECT_TRUE(std::isfinite(bruteForceData.getCoulombEnergy()));
    EXPECT_TRUE(std::isfinite(bruteForceData.getNonCoulombEnergy()));
    EXPECT_TRUE(std::isfinite(cellListData.getCoulombEnergy()));
    EXPECT_TRUE(std::isfinite(cellListData.getNonCoulombEnergy()));
}

TEST(PotentialTemplates, QmChargesAndOneWayInteractions)
{
    PotentialSettings::setCoulombRadiusCutOff(kCutOff);
    CoulombPotential::setCoulombRadiusCutOff(kCutOff);
    CoulombPotential::setCoulombEnergyCutOff(0.0);
    CoulombPotential::setCoulombForceCutOff(0.0);

    PotentialBruteForce potential;
    potential.makeCoulombPotential(CoulombShiftedPotential(kCutOff));
    potential.setNonCoulombPotential(makeNonCoulombPotential());

    simulationBox::OrthorhombicBox box;
    box.setBoxDimensions({15.0, 15.0, 15.0});

    Molecule mol1;
    mol1.setMoltype(1);
    Molecule mol2;
    mol2.setMoltype(2);

    Atom atom1;
    atom1.setPosition({0.0, 0.0, 0.0});
    atom1.setPartialCharge(-0.8);
    atom1.setQMCharge(-0.9);
    atom1.setAtomType(0);
    atom1.setInternalGlobalVDWType(VdwType{0});
    atom1.setForceToZero();

    Atom atom2;
    atom2.setPosition({1.2, 0.1, 0.0});
    atom2.setPartialCharge(0.4);
    atom2.setAtomType(0);
    atom2.setInternalGlobalVDWType(VdwType{0});
    atom2.setForceToZero();

    HybridSettings::setUseQMCharges(true);
    const auto coulombEnergy =
        potential.calculateSingleCoulombInteraction<QMChargeTag, MMChargeTag>(
            box,
            atom1,
            atom2
        );
    EXPECT_NE(coulombEnergy, 0.0);
    EXPECT_EQ(atom1.getForce(), -atom2.getForce());

    atom1.setForceToZero();
    atom2.setForceToZero();
    const auto energies =
        potential.calculateSingleInteractionOneWay<QMChargeTag, MMChargeTag>(
            box,
            mol1,
            mol2,
            atom1,
            atom2
        );
    EXPECT_NE(energies.first, 0.0);
    EXPECT_NE(energies.second, 0.0);
    EXPECT_NE(atom1.getForce(), Vec3D{});
    EXPECT_EQ(atom2.getForce(), Vec3D{});

    HybridSettings::setUseQMCharges(false);
    EXPECT_DOUBLE_EQ(potential.getPartialCharge<QMChargeTag>(atom1), -0.8);
    EXPECT_DOUBLE_EQ(potential.getPartialCharge<MMChargeTag>(atom2), 0.4);
}

TEST(PotentialStrategies, HybridRegionsExerciseBruteForceAndCellList)
{
    Settings::setJobtype(JobType::QMMM_MD);
    HybridSettings::setUseQMCharges(true);
    PotentialSettings::setCoulombRadiusCutOff(kCutOff);
    PotentialSettings::setNonCoulombRadiusCutOff(kCutOff);
    CoulombPotential::setCoulombRadiusCutOff(kCutOff);
    CoulombPotential::setCoulombEnergyCutOff(0.0);
    CoulombPotential::setCoulombForceCutOff(0.0);
    WaterModelSettings::setIsInterWaterModelSet(false);

    auto                simBoxBruteForce = makeHybridWaterBox();
    CellList            unusedCellList;
    PhysicalData        bruteForceData;
    PotentialBruteForce bruteForce;
    bruteForce.makeCoulombPotential(CoulombShiftedPotential(kCutOff));
    bruteForce.setNonCoulombPotential(makeNonCoulombPotential());

    bruteForce
        .calculateForces(simBoxBruteForce, bruteForceData, unusedCellList);
    resetForces(simBoxBruteForce);
    bruteForce
        .calculateQMMMForces(simBoxBruteForce, bruteForceData, unusedCellList);
    bruteForce.calculateHotspotSmoothingMMForces(
        simBoxBruteForce,
        bruteForceData,
        unusedCellList
    );

    auto              simBoxCellList = makeHybridWaterBox();
    auto              cellList       = makeCellList(simBoxCellList);
    PhysicalData      cellListData;
    PotentialCellList cellListPotential;
    cellListPotential.makeCoulombPotential(CoulombShiftedPotential(kCutOff));
    cellListPotential.setNonCoulombPotential(makeNonCoulombPotential());

    cellListPotential.calculateForces(simBoxCellList, cellListData, cellList);
    resetForces(simBoxCellList);
    cellListPotential
        .calculateQMMMForces(simBoxCellList, cellListData, cellList);
    cellListPotential.calculateHotspotSmoothingMMForces(
        simBoxCellList,
        cellListData,
        cellList
    );

    EXPECT_TRUE(std::isfinite(bruteForceData.getCoulombEnergy()));
    EXPECT_TRUE(std::isfinite(bruteForceData.getNonCoulombEnergy()));
    EXPECT_TRUE(std::isfinite(cellListData.getCoulombEnergy()));
    EXPECT_TRUE(std::isfinite(cellListData.getNonCoulombEnergy()));
    EXPECT_NE(bruteForce.clone(), nullptr);
    EXPECT_NE(cellListPotential.clone(), nullptr);

    WaterModelSettings::setIsInterWaterModelSet(true);
    auto         filteredBox      = makeHybridWaterBox();
    auto         filteredCellList = makeCellList(filteredBox);
    PhysicalData filteredData;
    cellListPotential
        .calculateQMMMForces(filteredBox, filteredData, filteredCellList);
    cellListPotential.calculateHotspotSmoothingMMForces(
        filteredBox,
        filteredData,
        filteredCellList
    );

    EXPECT_TRUE(std::isfinite(filteredData.getCoulombEnergy()));
    EXPECT_TRUE(std::isfinite(filteredData.getNonCoulombEnergy()));
}

TEST(SimulationBoxViews, ConstAndMutableWaterViewsFilterCorrectly)
{
    constexpr WaterGeometry geometry{.oh1 = 0.96, .oh2 = 0.96, .angle = 1.82};
    SimulationBox           simBox;
    simBox.setBoxDimensions({15.0, 15.0, 15.0});
    simBox.setWaterType(kWaterType);
    addWater(simBox, {-2.0, 0.0, 0.0}, geometry, HybridZone::OUTER);
    addWater(simBox, {0.0, 0.0, 0.0}, geometry, HybridZone::CORE, false);
    addWater(simBox, {2.0, 0.0, 0.0}, geometry, HybridZone::OUTER, true, 2);

    auto mutableOutsideView = simBox.getMoleculesOutsideZone(HybridZone::CORE);
    auto mutableOutsideIt   = mutableOutsideView.begin();
    const auto mutableOutsideEnd = mutableOutsideView.end();

    EXPECT_TRUE(mutableOutsideIt != mutableOutsideEnd);
    EXPECT_EQ(mutableOutsideEnd - mutableOutsideIt, 2);

    size_t mutableOutside = 0;
    while (mutableOutsideIt != mutableOutsideEnd)
    {
        ++mutableOutside;
        ++mutableOutsideIt;
    }
    EXPECT_FALSE(mutableOutsideIt != mutableOutsideEnd);
    EXPECT_EQ(mutableOutside, 2);

    size_t mutableInactive = 0;
    for ([[maybe_unused]] auto &molecule : simBox.getInactiveMolecules())
        ++mutableInactive;
    EXPECT_EQ(mutableInactive, 1);

    size_t mutableWater = 0;
    for ([[maybe_unused]] auto &molecule : simBox.getWaterTypeMolecules())
        ++mutableWater;
    EXPECT_EQ(mutableWater, 1);

    const SimulationBox &constBox   = simBox;
    const auto           activeView = constBox.getActiveMolecules();
    size_t               active     = 0;
    for ([[maybe_unused]] const auto &molecule : activeView) ++active;
    EXPECT_EQ(active, 2);

    const auto waterView = constBox.getWaterTypeMolecules();
    size_t     water     = 0;
    for ([[maybe_unused]] const auto &molecule : waterView) ++water;
    EXPECT_EQ(water, 1);
    EXPECT_EQ(
        simBox.getMolecule(0).getAtom(0).getAtomicNumber(),
        AtomNumber{8}
    );
}
