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

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "atom.hpp"
#include "berendsenManostat.hpp"
#include "constants/internalConversionFactors.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "manostatSettings.hpp"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "stochasticRescalingManostat.hpp"
#include "thermostatSettings.hpp"
#include "timingsSettings.hpp"
#include "triclinicBox.hpp"

namespace
{
    class ManostatRegression : public ::testing::Test
    {
       protected:
        molsys::SimulationBox      box;
        physicalData::PhysicalData data;

        void SetUp() override
        {
            settings::GeneralSettings::setVirialType(VirialType::MOLECULAR);
            settings::GeneralSettings::setRandomSeed(42);
            settings::GeneralSettings::setIsRandomSeedSet(true);
            settings::ManostatSettings::setIsotropy(Isotropy::ISOTROPIC);
            settings::ManostatSettings::setFixedAxis(FixedAxis::NONE);
            settings::TimingsSettings::setTimeStep(1.0);
            settings::ThermostatSettings::setActualTargetTemperature(0.0);
            settings::PotentialSettings::setCoulombRadiusCutOff(0.1);
            box.setBoxDimensions({10.0, 10.0, 10.0});
            box.setVolume(box.calculateVolume());
            data.setVirial(linalg::tensor3D{0.0});
            data.setKineticEnergyMolecularVector(linalg::tensor3D{0.0});
            data.setKineticEnergyAtomicVector(linalg::tensor3D{0.0});
        }

        void addMolecule(
            const std::vector<double>& positions,
            const double               speed = 0.0
        )
        {
            molsys::Molecule molecule;
            for (const auto x : positions)
            {
                auto atom = std::make_shared<molsys::Atom>();
                atom->setMass(1.0);
                atom->setPosition({x, 0.0, 0.0});
                atom->setVelocity({speed, 0.0, 0.0});
                molecule.addAtom(atom);
                box.addAtom(atom);
            }
            molecule.calculateCenterOfMass(box.getBox());
            box.addMolecule(molecule);
            box.calculateTotalMass();
            box.calculateDensity();
            box.calculateDegreesOfFreedom();
            data.setVolume(box.getVolume());
            data.setDensity(box.getDensity());
        }
    };
}   // namespace

TEST_F(ManostatRegression, triclinicHydrostaticBalancePreservesCell)
{
    auto cell = std::make_shared<molsys::TriclinicBox>();
    cell->setBoxAngles({90.0, 90.0, 60.0});
    cell->setBoxDimensions({10.0, 10.0, 10.0});
    cell->setVolume(cell->calculateVolume());
    box.setBox(*cell);
    addMolecule({0.0});
    addMolecule({1.0});
    data.setVirial(linalg::diagonalMatrix(box.getVolume() / PRESSURE_FACTOR));
    settings::ManostatSettings::setIsotropy(Isotropy::FULL_ANISOTROPIC);
    manostat::FullAnisotropicBerendsenManostat
        berendsen(1.0, 1.0, 0.03, FixedAxis::NONE);
    manostat::FullAnisotropicStochasticRescalingManostat
        stochastic(1.0, 1.0, 0.03, FixedAxis::NONE);
    for (auto* coupling :
         std::vector<manostat::Manostat*>{&berendsen, &stochastic})
    {
        const auto oldCell = box.getBox().getBoxMatrix();
        coupling->applyManostat(box, data);
        EXPECT_NEAR(data.getPressure(), 1.0, 1e-12);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                EXPECT_NEAR(
                    box.getBox().getBoxMatrix()[i][j],
                    oldCell[i][j],
                    1e-12
                );
    }
}

TEST_F(ManostatRegression, triclinicLengthPressureMatchesMolecularWork)
{
    auto cell = std::make_shared<molsys::TriclinicBox>();
    cell->setBoxAngles({90.0, 90.0, 60.0});
    cell->setBoxDimensions({10.0, 10.0, 10.0});
    cell->setVolume(cell->calculateVolume());
    box.setBox(*cell);
    settings::ManostatSettings::setFixedAxis(FixedAxis::YZ);
    data.setVirial(
        linalg::tensor3D{{2.0, -1.0, 0.5}, {4.0, -2.0, 1.0}, {6.0, -3.0, 1.5}}
    );
    manostat::Manostat pressure;
    pressure.calculatePressure(box, data);
    EXPECT_NEAR(
        data.getPressure(),
        0.5 * PRESSURE_FACTOR / box.getVolume(),
        1e-12
    );
    // r=(1,2,3), F=(2,-1,0.5): changing the first cell length
    // moves r_x by (1 - 2/sqrt(3)) times the fractional length change.
    EXPECT_NEAR(
        data.getCoupledPressure(),
        (2.0 - 4.0 / std::sqrt(3.0)) * PRESSURE_FACTOR / box.getVolume(),
        1e-12
    );
}

TEST_F(ManostatRegression, berendsenModesHaveSameHydrostaticVolumeResponse)
{
    constexpr double increment = 1e-6;
    for (const auto fixed : {FixedAxis::NONE, FixedAxis::Z})
    {
        manostat::BerendsenManostat iso(1.0, 1.0, increment, fixed);
        manostat::SemiIsotropicBerendsenManostat
            semi(1.0, 1.0, increment, Isotropy::SEMI_ISOTROPIC_XY, fixed);
        manostat::AnisotropicBerendsenManostat
            aniso(1.0, 1.0, increment, fixed);
        manostat::FullAnisotropicBerendsenManostat
            full(1.0, 1.0, increment, fixed);
        for (auto* coupling : std::vector<manostat::BerendsenManostat*>{
                 &iso,
                 &semi,
                 &aniso,
                 &full
             })
        {
            coupling->calculatePressure(box, data);
            const auto mu = coupling->calculateMu();
            EXPECT_NEAR((1.0 - det(mu)) / increment, 1.0, 1e-6);
            if (fixed == FixedAxis::Z)
                EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
        }
    }
}

TEST_F(ManostatRegression, molecularCouplingRejectsAtomicVirial)
{
    addMolecule({-0.2, 0.2});
    settings::GeneralSettings::setVirialType(VirialType::ATOMIC);
    manostat::BerendsenManostat berendsen(1.0, 1.0, 0.03, FixedAxis::NONE);
    manostat::StochasticRescalingManostat
        stochastic(1.0, 1.0, 0.03, FixedAxis::NONE);
    for (auto* coupling :
         std::vector<manostat::Manostat*>{&berendsen, &stochastic})
    {
        EXPECT_THROW(
            coupling->applyManostat(box, data),
            exc::ManostatException
        );
        EXPECT_DOUBLE_EQ(box.getVolume(), 1000.0);
        EXPECT_DOUBLE_EQ(box.getAtom(0).getPosition()[0], -0.2);
    }
    manostat::Manostat reportPressure;
    EXPECT_NO_THROW(reportPressure.applyManostat(box, data));
}

TEST_F(ManostatRegression, invalidScalingPreservesSimulationState)
{
    manostat::SemiIsotropicBerendsenManostat negativeRoot(
        1.0,
        1.0,
        2.0,
        Isotropy::SEMI_ISOTROPIC_XY,
        FixedAxis::NONE
    );
    manostat::BerendsenManostat negativeVolume(1.0, 1.0, 2.0, FixedAxis::NONE);
    manostat::StochasticRescalingManostat
        overflow(-3000.0, 1.0, 1.0, FixedAxis::NONE);
    manostat::StochasticRescalingManostat
                                underflow(3000.0, 1.0, 1.0, FixedAxis::NONE);
    manostat::BerendsenManostat cutoff(1.0, 1.0, 0.875, FixedAxis::NONE);
    const auto                  proposals = std::array<manostat::Manostat*, 5>{
        &negativeRoot,
        &negativeVolume,
        &overflow,
        &underflow,
        &cutoff
    };
    for (size_t i = 0; i < proposals.size(); ++i)
    {
        SCOPED_TRACE(i);
        box = molsys::SimulationBox{};
        box.setBoxDimensions({10.0, 10.0, 10.0});
        box.setVolume(box.calculateVolume());
        addMolecule({4.95, -4.85}, 1.0);
        const auto oldCenter  = box.getMolecule(0).getCenterOfMass();
        const auto oldDensity = box.getDensity();
        settings::ManostatSettings::setIsotropy(
            i == 0 ? Isotropy::SEMI_ISOTROPIC_XY : Isotropy::ISOTROPIC
        );
        settings::PotentialSettings::setCoulombRadiusCutOff(i == 4 ? 3.0 : 0.1);
        EXPECT_THROW(
            proposals[i]->applyManostat(box, data),
            exc::ManostatException
        );
        EXPECT_EQ(box.getBoxDimensions(), linalg::Vec3D(10.0));
        EXPECT_DOUBLE_EQ(box.getVolume(), 1000.0);
        EXPECT_DOUBLE_EQ(box.getDensity(), oldDensity);
        EXPECT_DOUBLE_EQ(data.getVolume(), 1000.0);
        EXPECT_DOUBLE_EQ(data.getDensity(), oldDensity);
        EXPECT_EQ(box.getMolecule(0).getCenterOfMass(), oldCenter);
        EXPECT_DOUBLE_EQ(box.getAtom(0).getPosition()[0], 4.95);
        EXPECT_DOUBLE_EQ(box.getAtom(1).getPosition()[0], -4.85);
        EXPECT_EQ(box.getAtom(0).getVelocity(), linalg::Vec3D(1.0, 0.0, 0.0));
    }
}
