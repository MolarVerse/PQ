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

#include "testManostat.hpp"

#include <cmath>    // for pow
#include <memory>   // for make_shared, __shared_ptr_access
#include <string>   // for string, allocator

#include "atom.hpp"                                  // for Atom
#include "berendsenManostat.hpp"                     // for BerendsenManostat
#include "constants/internalConversionFactors.hpp"   // for _PRESSURE_FACTOR_
#include "exceptions.hpp"                            // for ManostatException
#include "gtest/gtest.h"           // for Message, TestPartResult
#include "mathUtilities.hpp"       // for compare
#include "molecule.hpp"            // for Molecule
#include "potentialSettings.hpp"   // for PotentialSettings
#include "stochasticRescalingManostat.hpp"   // for StochasticRescalingManostat
#include "thermostatSettings.hpp"            // for ThermostatSettings
#include "throwWithMessage.hpp"    // for EXPECT_THROW_MSG
#include "timingsSettings.hpp"     // for TimingsSettings
#include "vector3d.hpp"            // for Vector3D, Vec3D

class TestableStochasticRescalingManostat
    : public manostat::StochasticRescalingManostat
{
  public:
    using StochasticRescalingManostat::StochasticRescalingManostat;

    void setPressure(const double pressure) { _pressure = pressure; }
};

/**
 * @brief tests function calculate pressure
 *
 */
TEST_F(TestManostat, CalculatePressure)
{
    _manostat->calculatePressure(*_box, *_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * constants::_PRESSURE_FACTOR_);
}

/**
 * @brief tests function to change virial to atomic
 *
 */
TEST_F(TestManostat, ChangeVirialToAtomic)
{
    _data->changeKineticVirialToAtomic();

    _manostat->calculatePressure(*_box, *_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 2.0 * constants::_PRESSURE_FACTOR_);
}

/**
 * @brief tests application of berendsen manostat
 *
 */
TEST_F(TestManostat, testApplyBerendsenManostat)
{
    settings::PotentialSettings::setCoulombRadiusCutOff(0.99);
    _box->setBoxDimensions({2.0, 2.0, 2.0});
    const auto boxOld = _box->getBoxDimensions();

    auto       molecule = simulationBox::Molecule();
    const auto atom     = std::make_shared<simulationBox::Atom>();
    atom->setPosition({1.0, 0.0, 0.0});
    molecule.addAtom(atom);
    molecule.setCenterOfMass({1.0, 0.0, 0.0});
    molecule.setNumberOfAtoms(1);

    _box->addMolecule(molecule);

    settings::TimingsSettings::setTimeStep(0.5);
    _manostat = new manostat::BerendsenManostat(1.0, 0.1, 4.5);

    const auto scaleFactors = linearAlgebra::Vec3D(::pow(
        1.0 - 4.5 * 0.5 / 0.1 * (1.0 - 3.0 * constants::_PRESSURE_FACTOR_),
        1.0 / 3.0
    ));

    _manostat->applyManostat(*_box, *_data);
    auto boxNew = _box->getBoxDimensions();

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * constants::_PRESSURE_FACTOR_);
    EXPECT_NEAR(boxNew[0], (boxOld * scaleFactors)[0], 1e-8);
    EXPECT_NEAR(boxNew[1], (boxOld * scaleFactors)[1], 1e-8);
    EXPECT_NEAR(boxNew[2], (boxOld * scaleFactors)[2], 1e-8);
    EXPECT_TRUE(utilities::compare(
        _box->getMolecule(0).getAtomPosition(0),
        linearAlgebra::Vec3D(1.0, 0.0, 0.0) * scaleFactors,
        1e-9
    ));
}

/**
 * @brief tests application of berendsen manostat if coulomb radius is larger
 * than half of the minimum box dimension
 *
 */
TEST_F(
    TestManostat,
    testApplyBerendsenManostat_cutoffLargerThanHalfOfMinimumBoxDimension
)
{
    settings::PotentialSettings::setCoulombRadiusCutOff(10.0);
    _box->setBoxDimensions({2.0, 2.0, 2.0});

    settings::TimingsSettings::setTimeStep(0.5);
    _manostat = new manostat::BerendsenManostat(
        3.0 * constants::_PRESSURE_FACTOR_,
        0.1,
        4.5
    );

    EXPECT_THROW_MSG(
        _manostat->applyManostat(*_box, *_data),
        customException::ManostatException,
        "Coulomb radius cut off is larger than half of the minimal box "
        "dimension"
    );
}

/**
 * @brief tests application of manotstat none
 *
 */
TEST_F(TestManostat, applyNoneManostat)
{
    _manostat->applyManostat(*_box, *_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * constants::_PRESSURE_FACTOR_);
}

TEST_F(TestManostat, stochasticRescalingMuUsesLengthScaling)
{
    settings::ThermostatSettings::setActualTargetTemperature(0.0);
    settings::TimingsSettings::setTimeStep(0.5);

    auto manostat = TestableStochasticRescalingManostat(7.0, 0.25, 0.12);
    manostat.setPressure(1.0);

    const auto mu = manostat.calculateMu(10.0);
    const auto expected =
        ::exp(-(0.12 * 0.5 / 0.25) * (7.0 - 1.0) / 3.0);

    EXPECT_DOUBLE_EQ(mu[0][0], expected);
    EXPECT_DOUBLE_EQ(mu[1][1], expected);
    EXPECT_DOUBLE_EQ(mu[2][2], expected);
}

/**
 * @brief test rotation of mu
 */
TEST_F(TestManostat, testRotateMu)
{
    auto mu = linearAlgebra::tensor3D({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
    });

    _manostat->rotateMu(mu);

    EXPECT_EQ(
        mu,
        linearAlgebra::tensor3D({
            {1.0, 6.0, 10.0},
            {0.0, 5.0, 14.0},
            {0.0, 0.0, 9.0},
        })
    );
}

/* ---------- BerendsenManostat — type, isotropy, getters ---------- */

TEST_F(TestManostat, berendsen_taueAndCompressibilityGetters)
{
    auto bm = manostat::BerendsenManostat(1.0, 0.1, 4.5);
    EXPECT_DOUBLE_EQ(bm.getTau(), 0.1);
    EXPECT_DOUBLE_EQ(bm.getCompressibility(), 4.5);
}

TEST_F(TestManostat, berendsen_manostatType)
{
    auto bm = manostat::BerendsenManostat(1.0, 0.1, 4.5);
    EXPECT_EQ(bm.getManostatType(), settings::ManostatType::BERENDSEN);
}

TEST_F(TestManostat, berendsen_isotropy)
{
    auto bm = manostat::BerendsenManostat(1.0, 0.1, 4.5);
    EXPECT_EQ(bm.getIsotropy(), settings::Isotropy::ISOTROPIC);
}

TEST_F(TestManostat, semiIsotropicBerendsen_isotropy)
{
    auto bm = manostat::SemiIsotropicBerendsenManostat(
        1.0, 0.1, 4.5, 2u, std::vector<size_t>{0u, 1u}
    );
    EXPECT_EQ(bm.getIsotropy(), settings::Isotropy::SEMI_ISOTROPIC);
    EXPECT_EQ(bm.getManostatType(), settings::ManostatType::BERENDSEN);
}

TEST_F(TestManostat, anisotropicBerendsen_isotropy)
{
    auto bm = manostat::AnisotropicBerendsenManostat(1.0, 0.1, 4.5);
    EXPECT_EQ(bm.getIsotropy(), settings::Isotropy::ANISOTROPIC);
    EXPECT_EQ(bm.getManostatType(), settings::ManostatType::BERENDSEN);
}

TEST_F(TestManostat, fullAnisotropicBerendsen_isotropy)
{
    auto bm = manostat::FullAnisotropicBerendsenManostat(1.0, 0.1, 4.5);
    EXPECT_EQ(bm.getIsotropy(), settings::Isotropy::FULL_ANISOTROPIC);
    EXPECT_EQ(bm.getManostatType(), settings::ManostatType::BERENDSEN);
}
