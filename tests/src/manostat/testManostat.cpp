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

#include <cmath>
#include <memory>

#include "atom.hpp"
#include "berendsenManostat.hpp"
#include "constants/internalConversionFactors.hpp"
#include "enums/manostat.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "manostatSettings.hpp"
#include "mathUtilities.hpp"
#include "molecule.hpp"
#include "potentialSettings.hpp"
#include "stochasticRescalingManostat.hpp"
#include "thermostatSettings.hpp"
#include "throwWithMessage.hpp"
#include "timingsSettings.hpp"
#include "triclinicBox.hpp"
#include "unit_types.hpp"
#include "units.hpp"

class TestableSimSpaceScalingManostat : public manostat::BerendsenManostat
{
   public:
    using BerendsenManostat::BerendsenManostat;

    static QuantityTensor3D toSimSpaceScaling(
        const linalg::tensor3D& scalingTensor,
        const molsys::Box&      box
    )
    {
        return _toSimSpaceScaling(scalingTensor, box);
    }
};

class TestableStochasticRescalingManostat
    : public manostat::StochasticRescalingManostat
{
   public:
    using StochasticRescalingManostat::StochasticRescalingManostat;

    void setPressure(const Pressure pressure) { _setPressure(pressure); }
    void setPressureTensor(const PressureTensor3D& pTensor)
    {
        _setPressureTensor(pTensor);
    }
};

class TestableAnisotropicStochasticRescalingManostat
    : public manostat::AnisotropicStochasticRescalingManostat
{
   public:
    using AnisotropicStochasticRescalingManostat::
        AnisotropicStochasticRescalingManostat;

    void setPressureTensor(const PressureTensor3D& pTensor)
    {
        _setPressureTensor(pTensor);
    }
};

class TestableSemiIsotropicStochasticRescalingManostat
    : public manostat::SemiIsotropicStochasticRescalingManostat
{
   public:
    using SemiIsotropicStochasticRescalingManostat::
        SemiIsotropicStochasticRescalingManostat;

    void setPressureTensor(const PressureTensor3D& pTensor)
    {
        _setPressureTensor(pTensor);
    }
};

class TestableFullAnisotropicStochasticRescalingManostat
    : public manostat::FullAnisotropicStochasticRescalingManostat
{
   public:
    using FullAnisotropicStochasticRescalingManostat::
        FullAnisotropicStochasticRescalingManostat;

    void setPressureTensor(const PressureTensor3D& pTensor)
    {
        _setPressureTensor(pTensor);
    }
};

class TestableBerendsenManostat : public manostat::BerendsenManostat
{
   public:
    using BerendsenManostat::BerendsenManostat;

    void setPressureTensor(const PressureTensor3D& pTensor)
    {
        _setPressureTensor(pTensor);
    }
};

class TestableSemiIsotropicBerendsenManostat
    : public manostat::SemiIsotropicBerendsenManostat
{
   public:
    using SemiIsotropicBerendsenManostat::SemiIsotropicBerendsenManostat;

    void setPressureTensor(const PressureTensor3D& pTensor)
    {
        _setPressureTensor(pTensor);
    }
};

class TestableAnisotropicBerendsenManostat
    : public manostat::AnisotropicBerendsenManostat
{
   public:
    using AnisotropicBerendsenManostat::AnisotropicBerendsenManostat;

    void setPressureTensor(const PressureTensor3D& pTensor)
    {
        _setPressureTensor(pTensor);
    }
};

class TestableFullAnisotropicBerendsenManostat
    : public manostat::FullAnisotropicBerendsenManostat
{
   public:
    using FullAnisotropicBerendsenManostat::FullAnisotropicBerendsenManostat;

    void setPressureTensor(const PressureTensor3D& pTensor)
    {
        _setPressureTensor(pTensor);
    }
};

namespace
{

    void setupCutMolecule(
        molsys::SimulationBox&      simulationBox,
        physicalData::PhysicalData& physicalData
    )
    {
        settings::PotentialSettings::setCoulombRadiusCutOff(4.0);
        settings::TimingsSettings::setTimeStep(1.0 * fs);

        molsys::OrthorhombicBox orthoBox{{10.0, 10.0, 10.0}};
        simulationBox.setBox(orthoBox, false);
        simulationBox.setTotalMass(2.0);

        physicalData.setVirial(diagonalMatrix(linalg::Vec3D(0.0)));
        physicalData.setKineticEnergyMolecularVector(
            diagonalMatrix(linalg::Vec3D(0.0))
        );
        physicalData.setKineticEnergyAtomicVector(
            diagonalMatrix(linalg::Vec3D(0.0))
        );

        auto atom1 = std::make_shared<molsys::Atom>();
        auto atom2 = std::make_shared<molsys::Atom>();

        atom1->setPosition({4.95, 0.0, 0.0});
        atom2->setPosition({-4.85, 0.0, 0.0});
        atom1->setMass(1.0);
        atom2->setMass(1.0);

        auto molecule = molsys::Molecule();
        molecule.addAtom(atom1);
        molecule.addAtom(atom2);
        molecule.calculateCenterOfMass(simulationBox.getBox());

        simulationBox.addAtom(atom1);
        simulationBox.addAtom(atom2);
        simulationBox.addMolecule(molecule);
    }

    linalg::Vec3D getMinimumImageDistance(molsys::SimulationBox& simulationBox)
    {
        const auto mol = simulationBox.getMolecule(0);

        auto dPosition = mol.getAtomPosition(AtomIndex{1}) -
                         mol.getAtomPosition(AtomIndex{0});
        simulationBox.applyPBC(dPosition);

        return dPosition;
    }

    void expectCutMoleculeScaled(molsys::SimulationBox& simulationBox)
    {
        const auto dPosition = getMinimumImageDistance(simulationBox);

        simulationBox.getMolecule(0).calculateCenterOfMass(simulationBox.getBox(
        ));
        const auto centerOfMass =
            simulationBox.getMolecule(0).getCenterOfMass();

        EXPECT_NEAR(simulationBox.getBoxDimensions()[0], 9.8, 1e-12);
        EXPECT_NEAR(centerOfMass[0], -4.851, 1e-12);
        EXPECT_NEAR(centerOfMass[1], 0.0, 1e-12);
        EXPECT_NEAR(centerOfMass[2], 0.0, 1e-12);
        EXPECT_NEAR(dPosition[0], 0.2, 1e-12);
        EXPECT_NEAR(dPosition[1], 0.0, 1e-12);
        EXPECT_NEAR(dPosition[2], 0.0, 1e-12);

        for (AtomIndex atomIndex{0}; atomIndex.get() < 2; ++atomIndex)
        {
            for (size_t axis = 0; axis < 3; ++axis)
            {
                const auto coordinate =
                    simulationBox.getMolecule(0).getAtomPosition(
                        atomIndex
                    )[axis];
                const auto halfBoxLength =
                    simulationBox.getBoxDimensions()[axis] / 2.0;

                EXPECT_GE(coordinate, -halfBoxLength);
                EXPECT_LT(coordinate, halfBoxLength);
            }
        }
    }

    double getMinimumImageDistance(
        molsys::SimulationBox& simulationBox,
        const size_t           moleculeIndex
    )
    {
        const auto mol = simulationBox.getMolecule(moleculeIndex);

        auto dPosition = mol.getAtomPosition(AtomIndex{1}) -
                         mol.getAtomPosition(AtomIndex{0});

        simulationBox.applyPBC(dPosition);

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

    EXPECT_DOUBLE_EQ(
        _data->getPressure().numerical_value_in(bar),
        3.0 * PRESSURE_FACTOR
    );
    EXPECT_DOUBLE_EQ(
        _data->getCoupledPressure().numerical_value_in(bar),
        3.0 * PRESSURE_FACTOR
    );
}

TEST_F(TestManostat, CalculatePressureWithFixedAxis)
{
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::Z);

    // _data has kinEnergyMolecular = diag(1, 2, 3), virial = diag(1, 2, 3)
    // 2 * kin + vir = diag(3, 6, 9) / volume(2.0) = diag(1.5, 3.0, 4.5) *
    // PRESSURE_FACTOR Total trace / 3 = 3.0 * PRESSURE_FACTOR Non-fixed (x, y)
    // avg = (1.5 + 3.0) / 2 = 2.25 * PRESSURE_FACTOR
    _manostat->calculatePressure(*_box, *_data);

    EXPECT_DOUBLE_EQ(
        _data->getPressure().numerical_value_in(bar),
        3.0 * PRESSURE_FACTOR
    );
    EXPECT_DOUBLE_EQ(
        _data->getCoupledPressure().numerical_value_in(bar),
        2.25 * PRESSURE_FACTOR
    );

    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::ALL);
    _manostat->calculatePressure(*_box, *_data);

    EXPECT_DOUBLE_EQ(
        _data->getCoupledPressure().numerical_value_in(bar),
        3.0 * PRESSURE_FACTOR
    );

    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::NONE);
}

/**
 * @brief tests function to change virial to atomic
 *
 */
TEST_F(TestManostat, ChangeVirialToAtomic)
{
    settings::GeneralSettings::setVirialType(VirialType::ATOMIC);
    _manostat->calculatePressure(*_box, *_data);

    EXPECT_DOUBLE_EQ(
        _data->getPressure().numerical_value_in(bar),
        2.0 * PRESSURE_FACTOR
    );

    // set virial type back to molecular for other tests
    settings::GeneralSettings::setVirialType(VirialType::MOLECULAR);
}

/**
 * @brief tests application of berendsen manostat
 *
 */
TEST_F(TestManostat, testApplyBerendsenManostat)
{
    settings::PotentialSettings::setCoulombRadiusCutOff(0.99);
    molsys::OrthorhombicBox orthoBox{{2.0, 2.0, 2.0}};
    _box->setBox(orthoBox, false);
    const auto boxOld = _box->getBoxDimensions();

    auto       molecule = molsys::Molecule();
    const auto atom     = std::make_shared<molsys::Atom>();
    atom->setPosition({1.0, 0.0, 0.0});
    molecule.addAtom(atom);
    molecule.setCenterOfMass({1.0, 0.0, 0.0});

    _box->addMolecule(molecule);

    settings::TimingsSettings::setTimeStep(0.5 * fs);
    _settings->manostatSettings.setTauByUser(0.1 * fs);
    _settings->manostatSettings.setCompressibilityByUser(4.5 * per_bar);
    _manostat = std::make_unique<manostat::BerendsenManostat>(
        _settings->manostatSettings
    );

    const auto scaleFactors = linalg::Vec3D(
        ::pow(
            1.0 - (4.5 * 0.5 / 0.1 * (1.0 - (0.75 * PRESSURE_FACTOR))),
            1.0 / 3.0
        )
    );

    _manostat->applyManostat(*_box, *_data);
    auto boxNew = _box->getBoxDimensions();

    EXPECT_DOUBLE_EQ(
        _data->getPressure().numerical_value_in(bar),
        0.75 * PRESSURE_FACTOR
    );
    EXPECT_NEAR(boxNew[0], (boxOld * scaleFactors)[0], 1e-8);
    EXPECT_NEAR(boxNew[1], (boxOld * scaleFactors)[1], 1e-8);
    EXPECT_NEAR(boxNew[2], (boxOld * scaleFactors)[2], 1e-8);
    EXPECT_TRUE(
        utilities::compare(
            _box->getMolecule(0).getAtomPosition(AtomIndex{0}),
            linalg::Vec3D(1.0, 0.0, 0.0) * scaleFactors,
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

    _settings->manostatSettings.setCompressibilityByUser(0.058808 * per_bar);
    auto manostat = std::make_unique<manostat::BerendsenManostat>(
        _settings->manostatSettings
    );
    manostat->applyManostat(*_box, *_data);

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
    ThermostatSettings::setActualTargetTemperature(toTemperature(0.0));

    _settings->manostatSettings.setTargetPressureByUser(
        -3.0 * ::log(0.98) * bar
    );
    _settings->manostatSettings.setCompressibilityByUser(1.0 * per_bar);
    auto manostat =
        manostat::StochasticRescalingManostat(_settings->manostatSettings);
    manostat.applyManostat(*_box, *_data);

    expectCutMoleculeScaled(*_box);
}

TEST_F(
    TestManostat,
    testApplyStochasticRescalingManostatMatchesCutAndInsideDistances
)
{
    setupCutMolecule(*_box, *_data);
    ThermostatSettings::setActualTargetTemperature(toTemperature(0.0));

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();

    atom1->setPosition({-1.0, 0.0, 0.0});
    atom2->setPosition({-0.8, 0.0, 0.0});
    atom1->setMass(1.0);
    atom2->setMass(1.0);

    auto molecule = molsys::Molecule();
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.calculateCenterOfMass(_box->getBox());

    _box->addAtom(atom1);
    _box->addAtom(atom2);
    _box->addMolecule(molecule);
    _box->setTotalMass(4.0);

    _settings->manostatSettings.setTargetPressureByUser(
        -3.0 * ::log(0.98) * bar
    );
    _settings->manostatSettings.setCompressibilityByUser(1.0 * per_bar);
    auto manostat =
        manostat::StochasticRescalingManostat(_settings->manostatSettings);
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
    molsys::OrthorhombicBox orthoBox{{2.0, 2.0, 2.0}};
    _box->setBox(orthoBox, false);

    _manostat->calculatePressure(*_box, *_data);
    settings::TimingsSettings::setTimeStep(0.5 * fs);
    _settings->manostatSettings.setTargetPressureByUser(_data->getPressure());
    _settings->manostatSettings.setTauByUser(0.1 * fs);
    _settings->manostatSettings.setCompressibilityByUser(4.5 * per_bar);
    _manostat = std::make_unique<manostat::BerendsenManostat>(
        _settings->manostatSettings
    );

    EXPECT_THROW_MSG(
        _manostat->applyManostat(*_box, *_data),
        exc::ManostatException,
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

    EXPECT_DOUBLE_EQ(
        _data->getPressure().numerical_value_in(bar),
        3.0 * PRESSURE_FACTOR
    );
}

TEST_F(TestManostat, stochasticRescalingMuUsesLengthScaling)
{
    ThermostatSettings::setActualTargetTemperature(toTemperature(0.0));
    settings::TimingsSettings::setTimeStep(0.5 * fs);

    _settings->manostatSettings.setTargetPressureByUser(7.0 * bar);
    _settings->manostatSettings.setTauByUser(0.25 * fs);
    _settings->manostatSettings.setCompressibilityByUser(0.12 * per_bar);
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::NONE);
    auto manostat =
        TestableStochasticRescalingManostat(_settings->manostatSettings);
    manostat.setPressure(1.0 * bar);

    const auto mu       = manostat.calculateMu(10.0).numerical_value_in(one);
    const auto expected = ::exp(-(0.12 * 0.5 / 0.25) * (7.0 - 1.0) / 3.0);

    EXPECT_DOUBLE_EQ(mu[0][0], expected);
    EXPECT_DOUBLE_EQ(mu[1][1], expected);
    EXPECT_DOUBLE_EQ(mu[2][2], expected);
}

TEST_F(TestManostat, stochasticRescalingPreservesInternalMolecularVelocities)
{
    _settings->manostatSettings.setIsotropyByUser(Isotropy::ISOTROPIC);
    settings::PotentialSettings::setCoulombRadiusCutOff(0.49);
    ThermostatSettings::setActualTargetTemperature(toTemperature(0.0));
    settings::TimingsSettings::setTimeStep(0.5 * fs);

    molsys::OrthorhombicBox orthoBox{{10.0, 10.0, 10.0}};
    _box->setBox(orthoBox, false);

    _data->setVirial(linalg::tensor3D(0.0));
    _data->setKineticEnergyMolecularVector(linalg::tensor3D(0.0));

    auto molecule = molsys::Molecule();

    const auto addAtom = [this, &molecule](
                             const linalg::Vec3D& position,
                             const linalg::Vec3D& velocity
                         )
    {
        auto atom = std::make_shared<molsys::Atom>();
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

    _settings->manostatSettings.setTargetPressureByUser(7.0 * bar);
    _settings->manostatSettings.setTauByUser(0.25 * fs);
    _settings->manostatSettings.setCompressibilityByUser(0.12 * per_bar);
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::NONE);
    _manostat = std::make_unique<manostat::StochasticRescalingManostat>(
        _settings->manostatSettings
    );

    const auto mu = ::exp(-(0.12 * 0.5 / 0.25) * (7.0 - 0.0) / 3.0);
    const auto expectedCenterOfMassVelocity = linalg::Vec3D(3.0 / mu, 0.0, 0.0);
    const auto expectedRelativeVelocity     = linalg::Vec3D(2.0, 0.0, 0.0);

    _manostat->applyManostat(*_box, *_data);

    const auto mol                  = _box->getMolecule(0);
    const auto velocity0            = mol.getAtomVelocity(AtomIndex{0});
    const auto velocity1            = mol.getAtomVelocity(AtomIndex{1});
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
    auto mu = QuantityTensor3D({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
    });

    manostat::Manostat::rotateMu(mu);

    EXPECT_EQ(
        mu,
        QuantityTensor3D({
            {1.0, 6.0, 10.0},
            {0.0, 5.0, 14.0},
            {0.0, 0.0, 9.0},
        })
    );
}

/* ---------- BerendsenManostat — type, isotropy, getters ---------- */

TEST_F(TestManostat, berendsenFixedAxesMu)
{
    settings::TimingsSettings::setTimeStep(0.5 * fs);

    // 1 fixed axis: X
    {
        ManostatSettings localSettings;
        localSettings.setTargetPressureByUser(1.0 * bar);
        localSettings.setTauByUser(0.5 * fs);
        localSettings.setCompressibilityByUser(0.2 * per_bar);
        localSettings.setFixedAxisByUser(FixedAxis::X);
        auto manostat = TestableBerendsenManostat(localSettings);
        manostat.setPressureTensor(
            diagonalMatrix(linalg::Vec3D(2.0, 3.0, 4.0)) * bar
        );

        const auto mu = manostat.calculateMu().numerical_value_in(one);
        EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
        EXPECT_DOUBLE_EQ(mu[1][1], ::sqrt(1.5));
        EXPECT_DOUBLE_EQ(mu[2][2], ::sqrt(1.5));
    }

    // 2 fixed axes: XY
    {
        ManostatSettings localSettings;
        localSettings.setTargetPressureByUser(1.0 * bar);
        localSettings.setTauByUser(0.5 * fs);
        localSettings.setCompressibilityByUser(0.2 * per_bar);
        localSettings.setFixedAxisByUser(FixedAxis::XY);
        auto manostat = TestableBerendsenManostat(localSettings);
        manostat.setPressureTensor(
            diagonalMatrix(linalg::Vec3D(2.0, 3.0, 4.0)) * bar
        );

        const auto mu = manostat.calculateMu().numerical_value_in(one);
        EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
        EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
        EXPECT_DOUBLE_EQ(mu[2][2], 1.6);
    }

    // All fixed axes
    {
        ManostatSettings localSettings;
        localSettings.setTargetPressureByUser(1.0 * bar);
        localSettings.setTauByUser(0.5 * fs);
        localSettings.setCompressibilityByUser(0.2 * per_bar);
        localSettings.setFixedAxisByUser(FixedAxis::ALL);
        auto manostat = TestableBerendsenManostat(localSettings);
        manostat.setPressureTensor(
            diagonalMatrix(linalg::Vec3D(2.0, 3.0, 4.0)) * bar
        );

        const auto mu = manostat.calculateMu().numerical_value_in(one);
        EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
        EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
        EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
    }
}

TEST_F(TestManostat, anisotropicBerendsenFixedAxesMu)
{
    settings::TimingsSettings::setTimeStep(0.5 * fs);

    ManostatSettings localSettings;
    localSettings.setTargetPressureByUser(1.0 * bar);
    localSettings.setTauByUser(0.5 * fs);
    localSettings.setCompressibilityByUser(0.2 * per_bar);
    localSettings.setFixedAxisByUser(FixedAxis::XZ);
    auto manostat = TestableAnisotropicBerendsenManostat(localSettings);
    manostat.setPressureTensor(
        diagonalMatrix(linalg::Vec3D(2.0, 3.0, 4.0)) * bar
    );

    const auto mu = manostat.calculateMu().numerical_value_in(one);
    EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
    EXPECT_DOUBLE_EQ(mu[1][1], 1.4);
    EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
}

TEST_F(TestManostat, fullAnisotropicBerendsenFixedAxesMu)
{
    settings::TimingsSettings::setTimeStep(0.5 * fs);

    ManostatSettings localSettings;
    localSettings.setTargetPressureByUser(1.0 * bar);
    localSettings.setTauByUser(0.5 * fs);
    localSettings.setCompressibilityByUser(0.2 * per_bar);
    localSettings.setFixedAxisByUser(FixedAxis::Y);
    auto manostat = TestableFullAnisotropicBerendsenManostat(localSettings);
    const auto pTensor =
        linalg::tensor3D({{2.0, 0.5, 0.1}, {0.5, 3.0, 0.2}, {0.1, 0.2, 4.0}});
    manostat.setPressureTensor(pTensor * bar);

    const auto mu = manostat.calculateMu().numerical_value_in(one);
    // Y row and column should be zeroed except diagonal which is 1.0
    EXPECT_DOUBLE_EQ(mu[1][0], 0.0);
    EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
    EXPECT_DOUBLE_EQ(mu[1][2], 0.0);
    EXPECT_DOUBLE_EQ(mu[0][1], 0.0);
    EXPECT_DOUBLE_EQ(mu[2][1], 0.0);
}

TEST_F(TestManostat, stochasticRescalingFixedAxesMu)
{
    ThermostatSettings::setActualTargetTemperature(toTemperature(0.0));
    settings::TimingsSettings::setTimeStep(0.5 * fs);

    // 1 fixed axis: Z
    {
        ManostatSettings localSettings;
        localSettings.setTargetPressureByUser(7.0 * bar);
        localSettings.setTauByUser(0.25 * fs);
        localSettings.setCompressibilityByUser(0.12 * per_bar);
        localSettings.setFixedAxisByUser(FixedAxis::Z);
        auto manostat = TestableStochasticRescalingManostat(localSettings);
        manostat.setPressureTensor(
            diagonalMatrix(linalg::Vec3D(1.0, 2.0, 3.0)) * bar
        );

        const auto mu = manostat.calculateMu(10.0).numerical_value_in(one);
        const auto expected = ::exp(-(0.12 * 0.5 / 0.25) * (7.0 - 1.5) / 2.0);

        EXPECT_DOUBLE_EQ(mu[0][0], expected);
        EXPECT_DOUBLE_EQ(mu[1][1], expected);
        EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
    }

    // 2 fixed axes: XY
    {
        ManostatSettings localSettings;
        localSettings.setTargetPressureByUser(7.0 * bar);
        localSettings.setTauByUser(0.25 * fs);
        localSettings.setCompressibilityByUser(0.12 * per_bar);
        localSettings.setFixedAxisByUser(FixedAxis::XY);
        auto manostat = TestableStochasticRescalingManostat(localSettings);
        manostat.setPressureTensor(
            diagonalMatrix(linalg::Vec3D(1.0, 2.0, 3.0)) * bar
        );

        const auto mu = manostat.calculateMu(10.0).numerical_value_in(one);
        const auto expected = ::exp(-(0.12 * 0.5 / 0.25) * (7.0 - 3.0) / 1.0);

        EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
        EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
        EXPECT_DOUBLE_EQ(mu[2][2], expected);
    }

    // All fixed axes
    {
        ManostatSettings localSettings;
        localSettings.setTargetPressureByUser(7.0 * bar);
        localSettings.setTauByUser(0.25 * fs);
        localSettings.setCompressibilityByUser(0.12 * per_bar);
        localSettings.setFixedAxisByUser(FixedAxis::ALL);
        auto manostat = TestableStochasticRescalingManostat(localSettings);
        manostat.setPressureTensor(
            diagonalMatrix(linalg::Vec3D(1.0, 2.0, 3.0)) * bar
        );

        const auto mu = manostat.calculateMu(10.0).numerical_value_in(one);

        EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
        EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
        EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
    }
}

TEST_F(TestManostat, anisotropicStochasticRescalingFixedAxesMu)
{
    ThermostatSettings::setActualTargetTemperature(toTemperature(0.0));
    settings::TimingsSettings::setTimeStep(0.5 * fs);

    ManostatSettings localSettings;
    localSettings.setTargetPressureByUser(7.0 * bar);
    localSettings.setTauByUser(0.25 * fs);
    localSettings.setCompressibilityByUser(0.12 * per_bar);
    localSettings.setFixedAxisByUser(FixedAxis::YZ);
    auto manostat =
        TestableAnisotropicStochasticRescalingManostat(localSettings);
    manostat.setPressureTensor(
        diagonalMatrix(linalg::Vec3D(1.0, 2.0, 3.0)) * bar
    );

    const auto mu       = manostat.calculateMu(10.0).numerical_value_in(one);
    const auto expected = ::exp(-(0.12 * 0.5 / 0.25) * (7.0 - 1.0) / 3.0);

    EXPECT_DOUBLE_EQ(mu[0][0], expected);
    EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
    EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
}

TEST_F(TestManostat, fullAnisotropicStochasticRescalingFixedAxesMu)
{
    ThermostatSettings::setActualTargetTemperature(toTemperature(0.0));
    settings::TimingsSettings::setTimeStep(0.5 * fs);

    ManostatSettings localSettings;
    localSettings.setTargetPressureByUser(7.0 * bar);
    localSettings.setTauByUser(0.25 * fs);
    localSettings.setCompressibilityByUser(0.12 * per_bar);
    localSettings.setFixedAxisByUser(FixedAxis::X);
    auto manostat =
        TestableFullAnisotropicStochasticRescalingManostat(localSettings);
    const auto pTensor =
        linalg::tensor3D({{1.0, 0.2, 0.3}, {0.2, 2.0, 0.4}, {0.3, 0.4, 3.0}});
    manostat.setPressureTensor(pTensor * bar);

    const auto mu = manostat.calculateMu(10.0).numerical_value_in(one);
    EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
    EXPECT_DOUBLE_EQ(mu[0][1], 0.0);
    EXPECT_DOUBLE_EQ(mu[0][2], 0.0);
    EXPECT_DOUBLE_EQ(mu[1][0], 0.0);
    EXPECT_DOUBLE_EQ(mu[2][0], 0.0);
}

TEST_F(TestManostat, semiIsotropicBerendsenFixedAnisotropicAxisMu)
{
    settings::TimingsSettings::setTimeStep(0.5 * fs);

    // xy isotropic (axes 0, 1), z anisotropic (axis 2) with Z fixed
    ManostatSettings localSettings;
    localSettings.setTargetPressureByUser(1.0 * bar);
    localSettings.setTauByUser(0.5 * fs);
    localSettings.setCompressibilityByUser(0.2 * per_bar);
    localSettings.setIsotropyByUser(Isotropy::SEMI_ISOTROPIC_XY);
    localSettings.setFixedAxisByUser(FixedAxis::Z);
    auto manostat = TestableSemiIsotropicBerendsenManostat(localSettings);
    manostat.setPressureTensor(
        diagonalMatrix(linalg::Vec3D(2.0, 4.0, 5.0)) * bar
    );

    const auto mu = manostat.calculateMu().numerical_value_in(one);
    // xy avg = 3.0, mu_xy = sqrt(1 - 0.2 * 0.5 / 0.5 * (1.0 - 3.0)) = sqrt(1.4)
    EXPECT_DOUBLE_EQ(mu[0][0], ::sqrt(1.4));
    EXPECT_DOUBLE_EQ(mu[1][1], ::sqrt(1.4));
    EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
}

TEST_F(TestManostat, semiIsotropicStochasticRescalingFixedAnisotropicAxisMu)
{
    ThermostatSettings::setActualTargetTemperature(toTemperature(0.0));
    settings::TimingsSettings::setTimeStep(0.5 * fs);

    // xz isotropic (axes 0, 2), y anisotropic (axis 1) with Y fixed
    ManostatSettings localSettings;
    localSettings.setTargetPressureByUser(7.0 * bar);
    localSettings.setTauByUser(0.25 * fs);
    localSettings.setCompressibilityByUser(0.12 * per_bar);
    localSettings.setIsotropyByUser(Isotropy::SEMI_ISOTROPIC_XZ);
    localSettings.setFixedAxisByUser(FixedAxis::Y);
    auto manostat =
        TestableSemiIsotropicStochasticRescalingManostat(localSettings);
    manostat.setPressureTensor(
        diagonalMatrix(linalg::Vec3D(1.0, 5.0, 3.0)) * bar
    );

    const auto mu = manostat.calculateMu(10.0).numerical_value_in(one);
    // xz avg = 2.0, deltaPxy = 7.0 - 2.0 = 5.0
    const auto expected_xz = ::exp(-(0.12 * 0.5 / 0.25) * 5.0 / 3.0);

    EXPECT_DOUBLE_EQ(mu[0][0], expected_xz);
    EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
    EXPECT_DOUBLE_EQ(mu[2][2], expected_xz);
}

/**
 * @brief orthorhombic boxes need no transformation of the scaling tensor
 */
TEST_F(TestManostat, toSimSpaceScalingOrthorhombicIsIdentityTransform)
{
    const auto mu = linalg::diagonalMatrix(linalg::Vec3D{0.9, 1.0, 1.1});
    const molsys::OrthorhombicBox box{{10.0, 11.0, 12.0}};

    const auto result =
        TestableSimSpaceScalingManostat::toSimSpaceScaling(mu, box);

    EXPECT_EQ(result, mu);
}

/**
 * @brief a box allowing a change of angles keeps the scaling tensor in
 * simulation space
 */
TEST_F(TestManostat, toSimSpaceScalingFullAnisotropicIsIdentityTransform)
{
    const linalg::tensor3D mu(
        {1.0, 0.1, 0.0},
        {0.0, 1.1, 0.2},
        {0.0, 0.0, 0.9}
    );
    const molsys::TriclinicBox box{{4.0, 5.0, 6.0}, {80.0, 75.0, 70.0}, true};

    const auto result =
        TestableSimSpaceScalingManostat::toSimSpaceScaling(mu, box);

    EXPECT_EQ(result, mu);
}

/**
 * @brief for fixed angles the diagonal scaling in orthogonal space is mapped
 * to simulation space: S_sim * (h v) = h * (S * v)
 */
TEST_F(TestManostat, toSimSpaceScalingTriclinicFixedAngles)
{
    const auto mu = linalg::diagonalMatrix(linalg::Vec3D{0.9, 1.0, 1.1});
    const molsys::TriclinicBox box{{4.0, 5.0, 6.0}, {80.0, 75.0, 70.0}, false};

    const auto muSim =
        TestableSimSpaceScalingManostat::toSimSpaceScaling(mu, box)
            .numerical_value_in(one);

    const linalg::Vec3D vector{0.3, -0.7, 1.9};
    const auto          expected = box.getBoxMatrix() * (mu * vector);
    const auto          result   = muSim * (box.getBoxMatrix() * vector);

    EXPECT_TRUE(utilities::compare(result, expected, 1e-12));

    // isotropic scaling is invariant under the transformation
    const auto isotropic = linalg::diagonalMatrix(linalg::Vec3D{0.95});
    const auto isotropicSim =
        TestableSimSpaceScalingManostat::toSimSpaceScaling(isotropic, box)
            .numerical_value_in(one);
    EXPECT_TRUE(
        utilities::compare(isotropicSim * vector, isotropic * vector, 1e-12)
    );
}
