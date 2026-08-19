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

#include "atom.hpp"                                  // for Atom
#include "berendsenManostat.hpp"                     // for BerendsenManostat
#include "constants/internalConversionFactors.hpp"   // for _PRESSURE_FACTOR_
#include "exceptions.hpp"                            // for ManostatException
#include "gtest/gtest.h"           // for Message, TestPartResult
#include "manostatSettings.hpp"    // for ManostatType, Isotropy
#include "mathUtilities.hpp"       // for compare
#include "molecule.hpp"            // for Molecule
#include "potentialSettings.hpp"   // for PotentialSettings
#include "settings.hpp"
#include "stochasticRescalingManostat.hpp"   // for StochasticRescalingManostat
#include "thermostatSettings.hpp"            // for ThermostatSettings
#include "throwWithMessage.hpp"              // for EXPECT_THROW_MSG
#include "timingsSettings.hpp"               // for TimingsSettings

class TestableStochasticRescalingManostat
    : public manostat::StochasticRescalingManostat
{
   public:
    using StochasticRescalingManostat::StochasticRescalingManostat;

    void setPressure(const double pressure) { _pressure = pressure; }
};

namespace
{
    void setupCutMolecule(
        simulationBox::SimulationBox& box,
        physicalData::PhysicalData&   data
    )
    {
        settings::PotentialSettings::setCoulombRadiusCutOff(4.0);
        settings::TimingsSettings::setTimeStep(1.0);

        box.setBoxDimensions({10.0, 10.0, 10.0});
        box.setVolume(box.calculateVolume());
        box.setTotalMass(2.0);

        data.setVirial(diagonalMatrix(linearAlgebra::Vec3D(0.0)));
        data.setKineticEnergyMolecularVector(
            diagonalMatrix(linearAlgebra::Vec3D(0.0))
        );
        data.setKineticEnergyAtomicVector(
            diagonalMatrix(linearAlgebra::Vec3D(0.0))
        );

        auto atom1 = std::make_shared<simulationBox::Atom>();
        auto atom2 = std::make_shared<simulationBox::Atom>();

        atom1->setPosition({4.95, 0.0, 0.0});
        atom2->setPosition({-4.85, 0.0, 0.0});
        atom1->setMass(1.0);
        atom2->setMass(1.0);

        auto molecule = simulationBox::Molecule();
        molecule.setNumberOfAtoms(2);
        molecule.setMolMass(2.0);
        molecule.addAtom(atom1);
        molecule.addAtom(atom2);
        molecule.calculateCenterOfMass(box.getBox());

        box.addAtom(atom1);
        box.addAtom(atom2);
        box.addMolecule(molecule);
    }

    linearAlgebra::Vec3D getMinimumImageDistance(
        simulationBox::SimulationBox& box
    )
    {
        auto dPosition = box.getMolecule(0).getAtomPosition(1) -
                         box.getMolecule(0).getAtomPosition(0);
        box.applyPBC(dPosition);

        return dPosition;
    }

    void expectCutMoleculeScaled(simulationBox::SimulationBox& box)
    {
        const auto dPosition = getMinimumImageDistance(box);

        box.getMolecule(0).calculateCenterOfMass(box.getBox());
        const auto centerOfMass = box.getMolecule(0).getCenterOfMass();

        EXPECT_NEAR(box.getBoxDimensions()[0], 9.8, 1e-12);
        EXPECT_NEAR(centerOfMass[0], -4.851, 1e-12);
        EXPECT_NEAR(centerOfMass[1], 0.0, 1e-12);
        EXPECT_NEAR(centerOfMass[2], 0.0, 1e-12);
        EXPECT_NEAR(dPosition[0], 0.2, 1e-12);
        EXPECT_NEAR(dPosition[1], 0.0, 1e-12);
        EXPECT_NEAR(dPosition[2], 0.0, 1e-12);

        for (size_t atomIndex = 0; atomIndex < 2; ++atomIndex)
        {
            for (size_t axis = 0; axis < 3; ++axis)
            {
                const auto coordinate =
                    box.getMolecule(0).getAtomPosition(atomIndex)[axis];
                const auto halfBoxLength = box.getBoxDimensions()[axis] / 2.0;

                EXPECT_GE(coordinate, -halfBoxLength);
                EXPECT_LT(coordinate, halfBoxLength);
            }
        }
    }

    double getMinimumImageDistance(
        simulationBox::SimulationBox& box,
        const size_t                  moleculeIndex
    )
    {
        auto dPosition = box.getMolecule(moleculeIndex).getAtomPosition(1) -
                         box.getMolecule(moleculeIndex).getAtomPosition(0);
        box.applyPBC(dPosition);

        return norm(dPosition);
    }
}   // namespace

/**
 * @brief tests function calculate pressure
 *
 */
TEST_F(TestManostat, CalculatePressure)
{
    _manostat->calculatePressure(*_box, *_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * constants::PRESSURE_FACTOR);
}

/**
 * @brief tests function to change virial to atomic
 *
 */
TEST_F(TestManostat, ChangeVirialToAtomic)
{
    settings::Settings::setVirialType(settings::VirialType::ATOMIC);
    _manostat->calculatePressure(*_box, *_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 2.0 * constants::PRESSURE_FACTOR);

    // set virial type back to molecular for other tests
    settings::Settings::setVirialType(settings::VirialType::MOLECULAR);
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
    _manostat = new manostat::BerendsenManostat(
        1.0,
        0.1,
        4.5,
        settings::FixedAxis::NONE
    );

    const auto scaleFactors = linearAlgebra::Vec3D(
        ::pow(
            1.0 - 4.5 * 0.5 / 0.1 * (1.0 - 3.0 * constants::PRESSURE_FACTOR),
            1.0 / 3.0
        )
    );

    _manostat->applyManostat(*_box, *_data);
    auto boxNew = _box->getBoxDimensions();

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * constants::PRESSURE_FACTOR);
    EXPECT_NEAR(boxNew[0], (boxOld * scaleFactors)[0], 1e-8);
    EXPECT_NEAR(boxNew[1], (boxOld * scaleFactors)[1], 1e-8);
    EXPECT_NEAR(boxNew[2], (boxOld * scaleFactors)[2], 1e-8);
    EXPECT_TRUE(
        utilities::compare(
            _box->getMolecule(0).getAtomPosition(0),
            linearAlgebra::Vec3D(1.0, 0.0, 0.0) * scaleFactors,
            1e-9
        )
    );
}

/**
 * @brief tests that manostat scaling keeps cut molecules internally intact
 *
 */
TEST_F(TestManostat, testApplyBerendsenManostatPreservesCutMoleculeGeometry)
{
    setupCutMolecule(*_box, *_data);

    auto manostat = manostat::BerendsenManostat(
        1.0,
        1.0,
        0.058808,
        settings::FixedAxis::NONE
    );
    manostat.applyManostat(*_box, *_data);

    expectCutMoleculeScaled(*_box);
}

/**
 * @brief tests that stochastic rescaling keeps cut molecules internally intact
 *
 */
TEST_F(
    TestManostat,
    testApplyStochasticRescalingManostatPreservesCutMoleculeGeometry
)
{
    setupCutMolecule(*_box, *_data);
    settings::ThermostatSettings::setActualTargetTemperature(0.0);

    auto manostat = manostat::StochasticRescalingManostat(
        -3.0 * ::log(0.98),
        1.0,
        1.0,
        settings::FixedAxis::NONE
    );
    manostat.applyManostat(*_box, *_data);

    expectCutMoleculeScaled(*_box);
}

TEST_F(
    TestManostat,
    testApplyStochasticRescalingManostatMatchesCutAndInsideDistances
)
{
    setupCutMolecule(*_box, *_data);
    settings::ThermostatSettings::setActualTargetTemperature(0.0);

    auto atom1 = std::make_shared<simulationBox::Atom>();
    auto atom2 = std::make_shared<simulationBox::Atom>();

    atom1->setPosition({-1.0, 0.0, 0.0});
    atom2->setPosition({-0.8, 0.0, 0.0});
    atom1->setMass(1.0);
    atom2->setMass(1.0);

    auto molecule = simulationBox::Molecule();
    molecule.setNumberOfAtoms(2);
    molecule.setMolMass(2.0);
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.calculateCenterOfMass(_box->getBox());

    _box->addAtom(atom1);
    _box->addAtom(atom2);
    _box->addMolecule(molecule);
    _box->setTotalMass(4.0);

    auto manostat = manostat::StochasticRescalingManostat(
        -3.0 * ::log(0.98),
        1.0,
        1.0,
        settings::FixedAxis::NONE
    );
    manostat.applyManostat(*_box, *_data);

    const auto cutDistance    = getMinimumImageDistance(*_box, 0);
    const auto insideDistance = getMinimumImageDistance(*_box, 1);

    EXPECT_NEAR(cutDistance, 0.2, 1e-12);
    EXPECT_NEAR(insideDistance, 0.2, 1e-12);
    EXPECT_NEAR(cutDistance, insideDistance, 1e-12);
}

/**
 * @brief tests application of berendsen manostat if coulomb radius is larger
 * than half of the minimum box dimension
 *
 */
TEST_F(
    TestManostat,
    testApplyBerendsenManostatCutoffLargerThanHalfOfMinimumBoxDimension
)
{
    settings::PotentialSettings::setCoulombRadiusCutOff(10.0);
    _box->setBoxDimensions({2.0, 2.0, 2.0});

    settings::TimingsSettings::setTimeStep(0.5);
    _manostat = new manostat::BerendsenManostat(
        3.0 * constants::PRESSURE_FACTOR,
        0.1,
        4.5,
        settings::FixedAxis::NONE
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

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * constants::PRESSURE_FACTOR);
}

TEST_F(TestManostat, stochasticRescalingMuUsesLengthScaling)
{
    settings::ThermostatSettings::setActualTargetTemperature(0.0);
    settings::TimingsSettings::setTimeStep(0.5);

    auto manostat = TestableStochasticRescalingManostat(
        7.0,
        0.25,
        0.12,
        settings::FixedAxis::NONE
    );
    manostat.setPressure(1.0);

    const auto mu       = manostat.calculateMu(10.0);
    const auto expected = ::exp(-(0.12 * 0.5 / 0.25) * (7.0 - 1.0) / 3.0);

    EXPECT_DOUBLE_EQ(mu[0][0], expected);
    EXPECT_DOUBLE_EQ(mu[1][1], expected);
    EXPECT_DOUBLE_EQ(mu[2][2], expected);
}

TEST_F(TestManostat, stochasticRescalingPreservesInternalMolecularVelocities)
{
    settings::ManostatSettings::setIsotropy(settings::Isotropy::ISOTROPIC);
    settings::PotentialSettings::setCoulombRadiusCutOff(0.49);
    settings::ThermostatSettings::setActualTargetTemperature(0.0);
    settings::TimingsSettings::setTimeStep(0.5);

    _box->setBoxDimensions({10.0, 10.0, 10.0});
    _box->setVolume(1000.0);

    _data->setVirial(linearAlgebra::tensor3D(0.0));
    _data->setKineticEnergyMolecularVector(linearAlgebra::tensor3D(0.0));

    auto molecule = simulationBox::Molecule();
    molecule.setNumberOfAtoms(2);
    molecule.setMolMass(2.0);

    const auto addAtom = [this, &molecule](
                             const linearAlgebra::Vec3D& position,
                             const linearAlgebra::Vec3D& velocity
                         )
    {
        auto atom = std::make_shared<simulationBox::Atom>();
        atom->setMass(1.0);
        atom->setPosition(position);
        atom->setVelocity(velocity);
        molecule.addAtom(atom);
        _box->addAtom(atom);
    };

    addAtom({1.0, 0.0, 0.0}, {2.0, 0.0, 0.0});
    addAtom({2.0, 0.0, 0.0}, {4.0, 0.0, 0.0});

    molecule.calculateCenterOfMass(_box->getBox());
    _box->addMolecule(molecule);

    _manostat = new manostat::StochasticRescalingManostat(
        7.0,
        0.25,
        0.12,
        settings::FixedAxis::NONE
    );

    const auto mu = ::exp(-(0.12 * 0.5 / 0.25) * (7.0 - 0.0) / 3.0);
    const auto expectedCenterOfMassVelocity =
        linearAlgebra::Vec3D(3.0 / mu, 0.0, 0.0);
    const auto expectedRelativeVelocity = linearAlgebra::Vec3D(2.0, 0.0, 0.0);

    _manostat->applyManostat(*_box, *_data);

    const auto velocity0            = _box->getMolecule(0).getAtomVelocity(0);
    const auto velocity1            = _box->getMolecule(0).getAtomVelocity(1);
    const auto centerOfMassVelocity = (velocity0 + velocity1) / 2.0;

    EXPECT_TRUE(
        utilities::compare(
            centerOfMassVelocity,
            expectedCenterOfMassVelocity,
            1e-12
        )
    );
    EXPECT_TRUE(
        utilities::compare(
            velocity1 - velocity0,
            expectedRelativeVelocity,
            1e-12
        )
    );
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

TEST_F(TestManostat, berendsenTauAndCompressibilityGetters)
{
    auto bm =
        manostat::BerendsenManostat(1.0, 0.1, 4.5, settings::FixedAxis::NONE);
    EXPECT_DOUBLE_EQ(bm.getTau(), 0.1);
    EXPECT_DOUBLE_EQ(bm.getCompressibility(), 4.5);
}

TEST_F(TestManostat, berendsenManostatType)
{
    auto bm =
        manostat::BerendsenManostat(1.0, 0.1, 4.5, settings::FixedAxis::NONE);
    EXPECT_EQ(bm.getManostatType(), settings::ManostatType::BERENDSEN);
}

TEST_F(TestManostat, berendsenIsotropy)
{
    auto bm =
        manostat::BerendsenManostat(1.0, 0.1, 4.5, settings::FixedAxis::NONE);
    EXPECT_EQ(bm.getIsotropy(), settings::Isotropy::ISOTROPIC);
}

TEST_F(TestManostat, semiIsotropicBerendsenIsotropy)
{
    auto bm = manostat::SemiIsotropicBerendsenManostat(
        1.0,
        0.1,
        4.5,
        2U,
        std::vector<size_t>{0U, 1U},
        settings::FixedAxis::NONE
    );
    EXPECT_EQ(bm.getIsotropy(), settings::Isotropy::SEMI_ISOTROPIC);
    EXPECT_EQ(bm.getManostatType(), settings::ManostatType::BERENDSEN);
}

TEST_F(TestManostat, anisotropicBerendsenIsotropy)
{
    auto bm = manostat::AnisotropicBerendsenManostat(
        1.0,
        0.1,
        4.5,
        settings::FixedAxis::NONE
    );
    EXPECT_EQ(bm.getIsotropy(), settings::Isotropy::ANISOTROPIC);
    EXPECT_EQ(bm.getManostatType(), settings::ManostatType::BERENDSEN);
}

TEST_F(TestManostat, fullAnisotropicBerendsenIsotropy)
{
    auto bm = manostat::FullAnisotropicBerendsenManostat(
        1.0,
        0.1,
        4.5,
        settings::FixedAxis::NONE
    );
    EXPECT_EQ(bm.getIsotropy(), settings::Isotropy::FULL_ANISOTROPIC);
    EXPECT_EQ(bm.getManostatType(), settings::ManostatType::BERENDSEN);
}
