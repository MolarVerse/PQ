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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <string>
#include <utility>

#include "SPCIntraWater.hpp"
#include "atom.hpp"
#include "celllist.hpp"
#include "coulombPotential.hpp"
#include "coulombShiftedPotential.hpp"
#include "exceptions.hpp"
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

namespace
{
    constexpr MolType kWaterType{1};
    constexpr double  kCutOff = 4.0;

    struct WaterGeometry
    {
        double oh1;
        double oh2;
        double angle;
    };

    void addWater(
        molsys::SimulationBox   &simulationBox,
        const linalg::Vec3D     &origin,
        const WaterGeometry     &geometry,
        const molsys::HybridZone zone,
        const bool               active,
        MolType                  molType
    )
    {
        const auto oxygen    = std::make_shared<molsys::Atom>();
        const auto hydrogen1 = std::make_shared<molsys::Atom>();
        const auto hydrogen2 = std::make_shared<molsys::Atom>();

        oxygen->setAtomicNumber(AtomNumber{8});
        oxygen->setPartialCharge(-0.82);
        oxygen->setQMCharge(-0.9);
        oxygen->setPosition(origin);
        oxygen->setAtomType(AtomType{0});
        oxygen->setInternalGlobalVDWType(VdwType{0});
        oxygen->setForceToZero();

        hydrogen1->setAtomicNumber(AtomNumber{1});
        hydrogen1->setPartialCharge(0.41);
        hydrogen1->setQMCharge(0.45);
        hydrogen1->setPosition(origin + linalg::Vec3D{geometry.oh1, 0.0, 0.0});
        hydrogen1->setAtomType(AtomType{1});
        hydrogen1->setInternalGlobalVDWType(VdwType{0});
        hydrogen1->setForceToZero();

        hydrogen2->setAtomicNumber(AtomNumber{1});
        hydrogen2->setPartialCharge(0.41);
        hydrogen2->setQMCharge(0.45);
        hydrogen2->setPosition(
            origin +
            linalg::Vec3D{
                geometry.oh2 * std::cos(geometry.angle),
                geometry.oh2 * std::sin(geometry.angle),
                0.0
            }
        );
        hydrogen2->setAtomType(AtomType{1});
        hydrogen2->setInternalGlobalVDWType(VdwType{0});
        hydrogen2->setForceToZero();

        molsys::Molecule water;
        water.setMoltype(molType);
        water.setHybridZone(zone);
        water.setSmoothingFactor(0.25);
        water.addAtom(oxygen);
        water.addAtom(hydrogen1);
        water.addAtom(hydrogen2);

        if (!active)
            water.deactivateMolecule();

        simulationBox.addAtom(oxygen);
        simulationBox.addAtom(hydrogen1);
        simulationBox.addAtom(hydrogen2);
        simulationBox.addMolecule(water);
    }

    void addWater(
        molsys::SimulationBox   &simulationBox,
        const linalg::Vec3D     &origin,
        const WaterGeometry     &geometry,
        const molsys::HybridZone zone
    )
    {
        addWater(simulationBox, origin, geometry, zone, true, kWaterType);
    }

    void addWater(
        molsys::SimulationBox   &simulationBox,
        const linalg::Vec3D     &origin,
        const WaterGeometry     &geometry,
        const molsys::HybridZone zone,
        bool                     active
    )
    {
        addWater(simulationBox, origin, geometry, zone, active, kWaterType);
    }

    molsys::SimulationBox makeIntraWaterBox(const WaterGeometry &geometry)
    {
        molsys::SimulationBox simBox;
        simBox.setBoxDimensions({20.0, 20.0, 20.0});
        simBox.setWaterType(kWaterType);
        addWater(
            simBox,
            {0.0, 0.0, 0.0},
            geometry,
            molsys::HybridZone::SMOOTHING
        );
        return simBox;
    }

    template <typename Model>
    void expectIntraModelConservesForce(
        Model               &model,
        const WaterGeometry &geometry
    )
    {
        auto                       simBox = makeIntraWaterBox(geometry);
        physicalData::PhysicalData data;

        model.calculate(simBox, data);

        const auto mol        = simBox.getMolecule(0);
        const auto totalForce = mol.getAtomForce(AtomIndex{0}) +
                                mol.getAtomForce(AtomIndex{1}) +
                                mol.getAtomForce(AtomIndex{2});

        EXPECT_NEAR(totalForce[0], 0.0, 1.0e-12);
        EXPECT_NEAR(totalForce[1], 0.0, 1.0e-12);
        EXPECT_NEAR(totalForce[2], 0.0, 1.0e-12);
        EXPECT_TRUE(std::isfinite(data.getBondEnergy()));
        EXPECT_TRUE(std::isfinite(data.getAngleEnergy()));
        EXPECT_GT(data.getBondEnergy(), 0.0);
    }

    /**
     * Near a singularity the forces are huge, so the net force is compared
     * with the largest single force, not with an absolute tolerance.
     */
    template <typename Model>
    void expectFiniteAndConservative(
        Model               &model,
        const WaterGeometry &geometry
    )
    {
        auto                       simBox = makeIntraWaterBox(geometry);
        physicalData::PhysicalData data;

        model.calculate(simBox, data);

        const auto mol     = simBox.getMolecule(0);
        auto       total   = linalg::Vec3D{0.0, 0.0, 0.0};
        auto       biggest = 1.0;

        for (std::size_t atom = 0; atom < 3; ++atom)
        {
            const auto force = mol.getAtomForce(AtomIndex{atom});
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                EXPECT_TRUE(std::isfinite(force[axis]))
                    << "atom " << atom << " axis " << axis;
                biggest = std::max(biggest, std::fabs(force[axis]));
            }
            total += force;
        }

        for (std::size_t axis = 0; axis < 3; ++axis)
            EXPECT_NEAR(total[axis], 0.0, 1.0e-9 * biggest);

        EXPECT_TRUE(std::isfinite(data.getBondEnergy()));
        EXPECT_TRUE(std::isfinite(data.getAngleEnergy()));
    }

    /**
     * The model must refuse a degenerate geometry with a descriptive
     * exception instead of silently producing NaN forces and energies.
     */
    template <typename Model>
    void expectDegenerateGeometryThrows(
        Model               &model,
        const WaterGeometry &geometry,
        const std::string   &expectedQuantity
    )
    {
        auto                       simBox = makeIntraWaterBox(geometry);
        physicalData::PhysicalData data;

        try
        {
            model.calculate(simBox, data);
            FAIL() << "no exception for a degenerate geometry ("
                   << expectedQuantity << ")";
        }
        catch (const exc::WaterModelException &error)
        {
            const std::string message = error.what();
            EXPECT_NE(
                message.find("Degenerate water geometry"),
                std::string::npos
            ) << message;
            EXPECT_NE(message.find(expectedQuantity), std::string::npos)
                << message;
        }
    }

    std::shared_ptr<pot::GuffNonCoulomb> makeNonCoulombPotential()
    {
        auto nonCoulomb = std::make_shared<pot::GuffNonCoulomb>();
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

        const auto pair = std::make_shared<pot::LennardJonesPair>(
            kCutOff,
            LJParams{.c6 = -1.0, .c12 = 1.0}
        );

        for (MolType mol1{1}; mol1.get() <= 2; ++mol1)
        {
            for (MolType mol2{1}; mol2.get() <= 2; ++mol2)
            {
                for (AtomType atom1{0}; atom1.get() < 2; ++atom1)
                {
                    for (AtomType atom2{0}; atom2.get() < 2; ++atom2)
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

    class ExposedInterWaterStrategy : public waterModel::InterWaterStrategy
    {
       public:
        void calculate(
            const waterModel::InterWaterState & /*state*/,
            molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*data*/,
            const std::shared_ptr<pot::CoulombPotential> & /*coulomb*/,
            const molsys::CellList & /*cellList*/
        ) final
        {
        }

        void calculateCoreToOuterForces(
            const waterModel::InterWaterState & /*state*/,
            molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*data*/,
            const std::shared_ptr<pot::CoulombPotential> & /*coulomb*/,
            const molsys::CellList & /*cellList*/
        ) final
        {
        }

        void calculateLayerToOuterForces(
            const waterModel::InterWaterState & /*state*/,
            molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*data*/,
            const std::shared_ptr<pot::CoulombPotential> & /*coulomb*/,
            const molsys::CellList & /*cellList*/
        ) final
        {
        }

        void calculateOuterToOuterForces(
            const waterModel::InterWaterState & /*state*/,
            molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*data*/,
            const std::shared_ptr<pot::CoulombPotential> & /*coulomb*/,
            const molsys::CellList & /*cellList*/
        ) final
        {
        }

        void calculateHotspotSmoothingMMForces(
            const waterModel::InterWaterState & /*state*/,
            molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*data*/,
            const std::shared_ptr<pot::CoulombPotential> & /*coulomb*/,
            const molsys::CellList & /*cellList*/
        ) final
        {
        }
    };

    molsys::SimulationBox makeHybridWaterBox()
    {
        constexpr WaterGeometry geometry{
            .oh1   = 0.96,
            .oh2   = 0.98,
            .angle = 1.82
        };

        molsys::SimulationBox simBox;
        simBox.setBoxDimensions({15.0, 15.0, 15.0});
        simBox.setWaterType(kWaterType);

        addWater(
            simBox,
            {-5.8, -5.5, -5.5},
            geometry,
            molsys::HybridZone::CORE,
            false
        );
        addWater(
            simBox,
            {-5.2, -3.5, -5.5},
            geometry,
            molsys::HybridZone::CORE,
            false,
            MolType{2}
        );
        addWater(
            simBox,
            {-4.2, -5.2, -5.2},
            geometry,
            molsys::HybridZone::LAYER,
            false
        );
        addWater(
            simBox,
            {-3.8, -3.2, -5.2},
            geometry,
            molsys::HybridZone::LAYER,
            false,
            MolType{2}
        );
        addWater(
            simBox,
            {-1.8, -5.0, -5.0},
            geometry,
            molsys::HybridZone::SMOOTHING
        );
        addWater(
            simBox,
            {-0.2, -4.8, -4.8},
            geometry,
            molsys::HybridZone::SMOOTHING
        );
        addWater(
            simBox,
            {-1.0, -3.0, -5.0},
            geometry,
            molsys::HybridZone::SMOOTHING,
            true,
            MolType{2}
        );
        addWater(
            simBox,
            {1.5, -4.6, -4.6},
            geometry,
            molsys::HybridZone::OUTER
        );
        addWater(
            simBox,
            {-3.5, -3.5, -3.5},
            geometry,
            molsys::HybridZone::OUTER,
            true,
            MolType{2}
        );

        return simBox;
    }

    molsys::CellList makeCellList(molsys::SimulationBox &simulationBox)
    {
        settings::Settings::activateCellList();

        molsys::CellList cellList;
        cellList.setNumberOfCells(3);
        cellList.resizeCells();
        cellList.setup(simulationBox);
        cellList.updateCellList(simulationBox);
        cellList.assignMoleculeHybridZoneIndices();
        cellList.assignWaterMoleculeIndices(simulationBox);
        return cellList;
    }

    void resetForces(molsys::SimulationBox &simulationBox)
    {
        for (auto &molecule : simulationBox.getMolecules())
            molecule.setAtomForcesToZero();
    }

}   // namespace

TEST(IntraWater, FlexibleSpcModelsProduceFiniteConservativeForces)
{
    settings::HybridSettings::setSmoothingMethod(SmoothingMethod::HOTSPOT);

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
    settings::HybridSettings::setSmoothingMethod(SmoothingMethod::HOTSPOT);
    expectIntraModelConservesForce(
        spcMtr,
        {.oh1 = 1.04, .oh2 = 0.97, .angle = 1.88}
    );
    EXPECT_DOUBLE_EQ(spcMtr.getEqOHDistance(), 1.0);
    EXPECT_DOUBLE_EQ(spcMtr.getEqHHDistance(), 1.632993162);

    waterModel::TIP3PMTRIntraWater tip3pMtr;
    settings::HybridSettings::setSmoothingMethod(SmoothingMethod::EXACT);
    expectIntraModelConservesForce(
        tip3pMtr,
        {.oh1 = 1.00, .oh2 = 0.93, .angle = 1.82}
    );
    EXPECT_DOUBLE_EQ(tip3pMtr.getEqOHDistance(), 0.9572);
    EXPECT_DOUBLE_EQ(tip3pMtr.getEqHHDistance(), 1.5139);
}

TEST(IntraWater, SpcModelsRejectDegenerateGeometry)
{
    settings::HybridSettings::setSmoothingMethod(SmoothingMethod::HOTSPOT);

    waterModel::SPCFwIntraWater  spcFw;
    waterModel::qSPCFwIntraWater qSpcFw;

    // a hydrogen on top of the oxygen: the bond force divides by a zero
    // distance
    expectDegenerateGeometryThrows(
        spcFw,
        {.oh1 = 0.0, .oh2 = 1.0, .angle = 1.9},
        "O-H1 distance"
    );
    expectDegenerateGeometryThrows(
        spcFw,
        {.oh1 = 1.0, .oh2 = 0.0, .angle = 1.9},
        "O-H2 distance"
    );
    expectDegenerateGeometryThrows(
        qSpcFw,
        {.oh1 = 0.0, .oh2 = 1.0, .angle = 1.9},
        "O-H1 distance"
    );
    expectDegenerateGeometryThrows(
        qSpcFw,
        {.oh1 = 1.0, .oh2 = 0.0, .angle = 1.9},
        "O-H2 distance"
    );

    // both hydrogens on the same ray from the oxygen (angle 0, also with
    // different bond lengths): the angle force divides by sin(angle) = 0
    expectDegenerateGeometryThrows(
        spcFw,
        {.oh1 = 1.0, .oh2 = 1.0, .angle = 0.0},
        "H-O-H angle"
    );
    expectDegenerateGeometryThrows(
        spcFw,
        {.oh1 = 1.0, .oh2 = 0.5, .angle = 0.0},
        "H-O-H angle"
    );
    expectDegenerateGeometryThrows(
        qSpcFw,
        {.oh1 = 1.0, .oh2 = 1.0, .angle = 0.0},
        "H-O-H angle"
    );
}

TEST(IntraWater, SpcModelsStayFiniteNextToTheDegenerateGeometries)
{
    settings::HybridSettings::setSmoothingMethod(SmoothingMethod::HOTSPOT);

    // a linear molecule is not degenerate: sin(pi) does not vanish in floating
    // point and the cross product is exactly zero
    waterModel::SPCFwIntraWater spcFw;
    expectFiniteAndConservative(
        spcFw,
        {.oh1 = 1.04, .oh2 = 0.97, .angle = std::numbers::pi}
    );

    // a tiny but non-zero angle and a very short bond are finite as well
    expectFiniteAndConservative(
        spcFw,
        {.oh1 = 1.04, .oh2 = 0.97, .angle = 1.0e-6}
    );
    expectFiniteAndConservative(
        spcFw,
        {.oh1 = 1.0e-6, .oh2 = 0.97, .angle = 1.9}
    );
}

TEST(IntraWater, MtrModelsRejectDegenerateGeometry)
{
    settings::HybridSettings::setSmoothingMethod(SmoothingMethod::HOTSPOT);

    waterModel::SPCMTRIntraWater   spcMtr;
    waterModel::TIP3PMTRIntraWater tip3pMtr;

    expectDegenerateGeometryThrows(
        spcMtr,
        {.oh1 = 0.0, .oh2 = 1.0, .angle = 1.9},
        "O-H1 distance"
    );
    expectDegenerateGeometryThrows(
        spcMtr,
        {.oh1 = 1.0, .oh2 = 0.0, .angle = 1.9},
        "O-H2 distance"
    );
    expectDegenerateGeometryThrows(
        tip3pMtr,
        {.oh1 = 0.0, .oh2 = 1.0, .angle = 1.9},
        "O-H1 distance"
    );
    expectDegenerateGeometryThrows(
        tip3pMtr,
        {.oh1 = 1.0, .oh2 = 0.0, .angle = 1.9},
        "O-H2 distance"
    );

    // equal bond lengths and angle 0 put the two hydrogens on top of each other
    expectDegenerateGeometryThrows(
        spcMtr,
        {.oh1 = 1.0, .oh2 = 1.0, .angle = 0.0},
        "H-H distance"
    );
    expectDegenerateGeometryThrows(
        tip3pMtr,
        {.oh1 = 0.9, .oh2 = 0.9, .angle = 0.0},
        "H-H distance"
    );
}

TEST(IntraWater, MtrModelsStayFiniteNextToTheDegenerateGeometries)
{
    settings::HybridSettings::setSmoothingMethod(SmoothingMethod::HOTSPOT);

    waterModel::SPCMTRIntraWater spcMtr;

    // angle 0 with different bond lengths: the hydrogens are apart (no divisor
    // vanishes)
    expectFiniteAndConservative(
        spcMtr,
        {.oh1 = 1.04, .oh2 = 0.97, .angle = 0.0}
    );
    // a linear molecule
    expectFiniteAndConservative(
        spcMtr,
        {.oh1 = 1.04, .oh2 = 0.97, .angle = std::numbers::pi}
    );
}

TEST(InterWater, PairEvaluatorsApplySymmetricAndOneWayForces)
{
    settings::PotentialSettings::setCoulombRadiusCutOff(kCutOff);
    pot::CoulombPotential::setCoulombRadiusCutOff(kCutOff);
    pot::CoulombPotential::setCoulombEnergyCutOff(0.0);
    pot::CoulombPotential::setCoulombForceCutOff(0.0);

    molsys::SimulationBox simBox;
    simBox.setBoxDimensions({15.0, 15.0, 15.0});

    molsys::Atom atom1;
    atom1.setPosition({0.0, 0.0, 0.0});
    atom1.setPartialCharge(-0.8);
    atom1.setQMCharge(-0.9);
    atom1.setForceToZero();

    molsys::Atom atom2;
    atom2.setPosition({1.2, 0.1, 0.0});
    atom2.setPartialCharge(0.4);
    atom2.setQMCharge(0.45);
    atom2.setForceToZero();

    const auto coulomb =
        std::make_shared<pot::CoulombShiftedPotential>(kCutOff);
    const pot::LennardJonesPair nonCoulomb(
        kCutOff,
        LJParams{.c6 = -1.0, .c12 = 1.0}
    );
    ExposedInterWaterStrategy strategy;

    EXPECT_DOUBLE_EQ(nonCoulomb.getRadialCutOff(), kCutOff);

    double coulombEnergy    = 0.0;
    double nonCoulombEnergy = 0.0;
    strategy.calculateSingleInteraction<pot::MMChargeTag, pot::MMChargeTag>(
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
    settings::HybridSettings::setUseQMCharges(true);
    coulombEnergy = 0.0;
    strategy
        .calculateSingleCoulombInteraction<pot::QMChargeTag, pot::MMChargeTag>(
            atom1,
            atom2,
            coulomb,
            kCutOff * kCutOff,
            simBox,
            coulombEnergy
        );
    EXPECT_NE(coulombEnergy, 0.0);
    EXPECT_EQ(atom1.getForce(), -atom2.getForce());
    EXPECT_DOUBLE_EQ(strategy.getPartialCharge<pot::QMChargeTag>(atom1), -0.9);

    atom1.setForceToZero();
    atom2.setForceToZero();
    coulombEnergy    = 0.0;
    nonCoulombEnergy = 0.0;
    strategy
        .calculateSingleInteractionOneWay<pot::MMChargeTag, pot::QMChargeTag>(
            atom1,
            atom2,
            coulomb,
            kCutOff * kCutOff,
            simBox,
            nonCoulomb,
            coulombEnergy,
            nonCoulombEnergy
        );
    EXPECT_NE(atom1.getForce(), linalg::Vec3D{});
    EXPECT_EQ(atom2.getForce(), linalg::Vec3D{});

    settings::HybridSettings::setUseQMCharges(false);
    EXPECT_DOUBLE_EQ(strategy.getPartialCharge<pot::QMChargeTag>(atom1), -0.8);
    EXPECT_DOUBLE_EQ(strategy.getPartialCharge<pot::MMChargeTag>(atom2), 0.4);
}

TEST(InterWater, DefaultStrategyIsInert)
{
    molsys::SimulationBox      simBox;
    physicalData::PhysicalData data;
    molsys::CellList           cellList;
    const auto                 coulomb =
        std::make_shared<pot::CoulombShiftedPotential>(kCutOff);

    waterModel::InterWater interWater;
    interWater.calculate(simBox, data, coulomb, cellList);
    interWater.calculateQMMMForces(simBox, data, coulomb, cellList);
    interWater
        .calculateHotspotSmoothingMMForces(simBox, data, coulomb, cellList);

    EXPECT_DOUBLE_EQ(data.getCoulombEnergy(), 0.0);
    EXPECT_DOUBLE_EQ(data.getNonCoulombEnergy(), 0.0);
}

TEST(InterWater, NonOxygenOnlyStateInitializesEveryPair)
{
    settings::PotentialSettings::setCoulombRadiusCutOff(kCutOff);
    settings::PotentialSettings::setNonCoulombRadiusCutOff(kCutOff);

    auto oxygenOxygen = std::make_unique<pot::LennardJonesPair>(
        kCutOff,
        LJParams{.c6 = -1.0, .c12 = 1.0}
    );
    auto oxygenHydrogen = std::make_unique<pot::LennardJonesPair>(
        kCutOff,
        LJParams{.c6 = -1.0, .c12 = 1.0}
    );
    auto hydrogenHydrogen = std::make_unique<pot::LennardJonesPair>(
        kCutOff,
        LJParams{.c6 = -1.0, .c12 = 1.0}
    );
    const auto *oxygenOxygenView     = oxygenOxygen.get();
    const auto *oxygenHydrogenView   = oxygenHydrogen.get();
    const auto *hydrogenHydrogenView = hydrogenHydrogen.get();

    waterModel::InterWaterState state;
    state.oxygenOnlyNonCoulomb = false;
    state.nonCoulombPairOO     = std::move(oxygenOxygen);
    state.nonCoulombPairOH     = std::move(oxygenHydrogen);
    state.nonCoulombPairHH     = std::move(hydrogenHydrogen);

    waterModel::InterWater interWater(
        std::move(state),
        std::make_unique<waterModel::InterWaterStrategyNull>()
    );
    waterModel::InterWater nullPairs(
        waterModel::InterWaterState{},
        std::make_unique<waterModel::InterWaterStrategyNull>()
    );
    molsys::SimulationBox      simBox;
    physicalData::PhysicalData physicalData;
    molsys::CellList           cellList;
    const auto                 coulomb =
        std::make_shared<pot::CoulombShiftedPotential>(kCutOff);

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
    settings::Settings::setJobtype(JobType::QMMM_MD);
    settings::HybridSettings::setUseQMCharges(true);
    settings::PotentialSettings::setCoulombRadiusCutOff(kCutOff);
    settings::PotentialSettings::setNonCoulombRadiusCutOff(kCutOff);
    pot::CoulombPotential::setCoulombRadiusCutOff(kCutOff);
    pot::CoulombPotential::setCoulombEnergyCutOff(0.0);
    pot::CoulombPotential::setCoulombForceCutOff(0.0);
    settings::WaterModelSettings::setIsInterWaterModelSet(true);

    const auto coulomb =
        std::make_shared<pot::CoulombShiftedPotential>(kCutOff);

    auto                       simBoxBruteForce = makeHybridWaterBox();
    molsys::CellList           unusedCellList;
    physicalData::PhysicalData bruteForceData;
    waterModel::InterWater     bruteForce(
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

    auto                       simBoxCellList = makeHybridWaterBox();
    auto                       cellList       = makeCellList(simBoxCellList);
    physicalData::PhysicalData cellListData;
    waterModel::InterWater     cellListWater(
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
    settings::PotentialSettings::setCoulombRadiusCutOff(kCutOff);
    pot::CoulombPotential::setCoulombRadiusCutOff(kCutOff);
    pot::CoulombPotential::setCoulombEnergyCutOff(0.0);
    pot::CoulombPotential::setCoulombForceCutOff(0.0);

    pot::PotentialBruteForce potential;
    potential.makeCoulombPotential(pot::CoulombShiftedPotential(kCutOff));
    potential.setNonCoulombPotential(makeNonCoulombPotential());

    molsys::OrthorhombicBox box;
    box.setBoxDimensions({15.0, 15.0, 15.0});

    molsys::Molecule mol1;
    mol1.setMoltype(MolType{1});
    molsys::Molecule mol2;
    mol2.setMoltype(MolType{2});

    molsys::Atom atom1;
    atom1.setPosition({0.0, 0.0, 0.0});
    atom1.setPartialCharge(-0.8);
    atom1.setQMCharge(-0.9);
    atom1.setAtomType(AtomType{0});
    atom1.setInternalGlobalVDWType(VdwType{0});
    atom1.setForceToZero();

    molsys::Atom atom2;
    atom2.setPosition({1.2, 0.1, 0.0});
    atom2.setPartialCharge(0.4);
    atom2.setAtomType(AtomType{0});
    atom2.setInternalGlobalVDWType(VdwType{0});
    atom2.setForceToZero();

    settings::HybridSettings::setUseQMCharges(true);
    const auto coulombEnergy = potential.calculateSingleCoulombInteraction<
        pot::QMChargeTag,
        pot::MMChargeTag>(box, atom1, atom2);
    EXPECT_NE(coulombEnergy, 0.0);
    EXPECT_EQ(atom1.getForce(), -atom2.getForce());

    atom1.setForceToZero();
    atom2.setForceToZero();
    const auto energies = potential.calculateSingleInteractionOneWay<
        pot::QMChargeTag,
        pot::MMChargeTag>(box, mol1, mol2, atom1, atom2);
    EXPECT_NE(energies.first, 0.0);
    EXPECT_NE(energies.second, 0.0);
    EXPECT_NE(atom1.getForce(), linalg::Vec3D{});
    EXPECT_EQ(atom2.getForce(), linalg::Vec3D{});

    settings::HybridSettings::setUseQMCharges(false);
    EXPECT_DOUBLE_EQ(potential.getPartialCharge<pot::QMChargeTag>(atom1), -0.8);
    EXPECT_DOUBLE_EQ(potential.getPartialCharge<pot::MMChargeTag>(atom2), 0.4);
}

TEST(PotentialStrategies, HybridRegionsExerciseBruteForceAndCellList)
{
    settings::Settings::setJobtype(JobType::QMMM_MD);
    settings::HybridSettings::setUseQMCharges(true);
    settings::PotentialSettings::setCoulombRadiusCutOff(kCutOff);
    settings::PotentialSettings::setNonCoulombRadiusCutOff(kCutOff);
    pot::CoulombPotential::setCoulombRadiusCutOff(kCutOff);
    pot::CoulombPotential::setCoulombEnergyCutOff(0.0);
    pot::CoulombPotential::setCoulombForceCutOff(0.0);
    settings::WaterModelSettings::setIsInterWaterModelSet(false);

    auto                       simBoxBruteForce = makeHybridWaterBox();
    molsys::CellList           unusedCellList;
    physicalData::PhysicalData bruteForceData;
    pot::PotentialBruteForce   bruteForce;
    bruteForce.makeCoulombPotential(pot::CoulombShiftedPotential(kCutOff));
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

    auto                       simBoxCellList = makeHybridWaterBox();
    auto                       cellList       = makeCellList(simBoxCellList);
    physicalData::PhysicalData cellListData;
    pot::PotentialCellList     cellListPotential;
    cellListPotential.makeCoulombPotential(
        pot::CoulombShiftedPotential(kCutOff)
    );
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

    settings::WaterModelSettings::setIsInterWaterModelSet(true);
    auto                       filteredBox      = makeHybridWaterBox();
    auto                       filteredCellList = makeCellList(filteredBox);
    physicalData::PhysicalData filteredData;
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
    molsys::SimulationBox   simBox;
    simBox.setBoxDimensions({15.0, 15.0, 15.0});
    simBox.setWaterType(kWaterType);
    addWater(simBox, {-2.0, 0.0, 0.0}, geometry, molsys::HybridZone::OUTER);
    addWater(
        simBox,
        {0.0, 0.0, 0.0},
        geometry,
        molsys::HybridZone::CORE,
        false
    );
    addWater(
        simBox,
        {2.0, 0.0, 0.0},
        geometry,
        molsys::HybridZone::OUTER,
        true,
        MolType{2}
    );

    auto mutableOutsideView =
        simBox.getMoleculesOutsideZone(molsys::HybridZone::CORE);
    auto       mutableOutsideIt  = mutableOutsideView.begin();
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

    const molsys::SimulationBox &constBox   = simBox;
    const auto                   activeView = constBox.getActiveMolecules();
    size_t                       active     = 0;
    for ([[maybe_unused]] const auto &molecule : activeView) ++active;
    EXPECT_EQ(active, 2);

    const auto waterView = constBox.getWaterTypeMolecules();
    size_t     water     = 0;
    for ([[maybe_unused]] const auto &molecule : waterView) ++water;
    EXPECT_EQ(water, 1);
    EXPECT_EQ(
        simBox.getMolecule(0).getAtom(AtomIndex{0}).getAtomicNumber(),
        AtomNumber{8}
    );
}
