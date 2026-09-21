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
// for Message, TestPartResult
#include "manostatSettings.hpp"    // for ManostatType, Isotropy
#include "mathUtilities.hpp"       // for compare
#include "molecule.hpp"            // for Molecule
#include "potentialSettings.hpp"   // for PotentialSettings
#include "settings.hpp"
#include "stochasticRescalingManostat.hpp"   // for StochasticRescalingManostat
#include "thermostatSettings.hpp"            // for ThermostatSettings
#include "throwWithMessage.hpp"              // for EXPECT_THROW_MSG
#include "timingsSettings.hpp"               // for TimingsSettings

namespace
{
    class TestableStochasticRescalingManostat
        : public manostat::StochasticRescalingManostat
    {
       public:
        using StochasticRescalingManostat::StochasticRescalingManostat;

        void setPressure(const double pressure) { _pressure = pressure; }
        void setPressureTensor(const linalg::tensor3D& pTensor)
        {
            _pressureTensor = pTensor;
        }
    };

    class TestableAnisotropicStochasticRescalingManostat
        : public manostat::AnisotropicStochasticRescalingManostat
    {
       public:
        using AnisotropicStochasticRescalingManostat::
            AnisotropicStochasticRescalingManostat;

        void setPressureTensor(const linalg::tensor3D& pTensor)
        {
            _pressureTensor = pTensor;
        }
    };

    class TestableSemiIsotropicStochasticRescalingManostat
        : public manostat::SemiIsotropicStochasticRescalingManostat
    {
       public:
        using SemiIsotropicStochasticRescalingManostat::
            SemiIsotropicStochasticRescalingManostat;

        void setPressureTensor(const linalg::tensor3D& pTensor)
        {
            _pressureTensor = pTensor;
        }
    };

    class TestableFullAnisotropicStochasticRescalingManostat
        : public manostat::FullAnisotropicStochasticRescalingManostat
    {
       public:
        using FullAnisotropicStochasticRescalingManostat::
            FullAnisotropicStochasticRescalingManostat;

        void setPressureTensor(const linalg::tensor3D& pTensor)
        {
            _pressureTensor = pTensor;
        }
    };

    class TestableBerendsenManostat : public manostat::BerendsenManostat
    {
       public:
        using BerendsenManostat::BerendsenManostat;

        void setPressureTensor(const linalg::tensor3D& pTensor)
        {
            _pressureTensor = pTensor;
        }
    };

    class TestableSemiIsotropicBerendsenManostat
        : public manostat::SemiIsotropicBerendsenManostat
    {
       public:
        using SemiIsotropicBerendsenManostat::SemiIsotropicBerendsenManostat;

        void setPressureTensor(const linalg::tensor3D& pTensor)
        {
            _pressureTensor = pTensor;
        }
    };

    class TestableAnisotropicBerendsenManostat
        : public manostat::AnisotropicBerendsenManostat
    {
       public:
        using AnisotropicBerendsenManostat::AnisotropicBerendsenManostat;

        void setPressureTensor(const linalg::tensor3D& pTensor)
        {
            _pressureTensor = pTensor;
        }
    };

    class TestableFullAnisotropicBerendsenManostat
        : public manostat::FullAnisotropicBerendsenManostat
    {
       public:
        using FullAnisotropicBerendsenManostat::
            FullAnisotropicBerendsenManostat;

        void setPressureTensor(const linalg::tensor3D& pTensor)
        {
            _pressureTensor = pTensor;
        }
    };

    void setupCutMolecule(
        molsys::SimulationBox&      simulationBox,
        physicalData::PhysicalData& physicalData
    )
    {
        settings::PotentialSettings::setCoulombRadiusCutOff(4.0);
        settings::TimingsSettings::setTimeStep(1.0);

        simulationBox.setBoxDimensions({10.0, 10.0, 10.0});
        simulationBox.setVolume(simulationBox.calculateVolume());
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
        molecule.setNumberOfAtoms(2);
        molecule.setMolMass(2.0);
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

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * PRESSURE_FACTOR);
    EXPECT_DOUBLE_EQ(_data->getCoupledPressure(), 3.0 * PRESSURE_FACTOR);
}

TEST_F(TestManostat, CalculatePressureWithFixedAxis)
{
    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::Z);

    // _data has kinEnergyMolecular = diag(1, 2, 3), virial = diag(1, 2, 3)
    // 2 * kin + vir = diag(3, 6, 9) / volume(2.0) = diag(1.5, 3.0, 4.5) *
    // PRESSURE_FACTOR Total trace / 3 = 3.0 * PRESSURE_FACTOR Non-fixed (x, y)
    // avg = (1.5 + 3.0) / 2 = 2.25 * PRESSURE_FACTOR
    _manostat->calculatePressure(*_box, *_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * PRESSURE_FACTOR);
    EXPECT_DOUBLE_EQ(_data->getCoupledPressure(), 2.25 * PRESSURE_FACTOR);

    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::ALL);
    _manostat->calculatePressure(*_box, *_data);

    EXPECT_DOUBLE_EQ(_data->getCoupledPressure(), 3.0 * PRESSURE_FACTOR);

    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::NONE);
}

/**
 * @brief tests function to change virial to atomic
 *
 */
TEST_F(TestManostat, ChangeVirialToAtomic)
{
    settings::Settings::setVirialType(settings::VirialType::ATOMIC);
    _manostat->calculatePressure(*_box, *_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 2.0 * PRESSURE_FACTOR);

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

    auto       molecule = molsys::Molecule();
    const auto atom     = std::make_shared<molsys::Atom>();
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

    const auto scaleFactors = linalg::Vec3D(
        ::pow(
            1.0 - (4.5 * 0.5 / 0.1 * (1.0 - (3.0 * PRESSURE_FACTOR))),
            1.0 / 3.0
        )
    );

    _manostat->applyManostat(*_box, *_data);
    auto boxNew = _box->getBoxDimensions();

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * PRESSURE_FACTOR);
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

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();

    atom1->setPosition({-1.0, 0.0, 0.0});
    atom2->setPosition({-0.8, 0.0, 0.0});
    atom1->setMass(1.0);
    atom2->setMass(1.0);

    auto molecule = molsys::Molecule();
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
        3.0 * PRESSURE_FACTOR,
        0.1,
        4.5,
        settings::FixedAxis::NONE
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

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * PRESSURE_FACTOR);
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

    _data->setVirial(linalg::tensor3D(0.0));
    _data->setKineticEnergyMolecularVector(linalg::tensor3D(0.0));

    auto molecule = molsys::Molecule();
    molecule.setNumberOfAtoms(2);
    molecule.setMolMass(2.0);

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

    _manostat = new manostat::StochasticRescalingManostat(
        7.0,
        0.25,
        0.12,
        settings::FixedAxis::NONE
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
    auto mu = linalg::tensor3D({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
    });

    manostat::Manostat::rotateMu(mu);

    EXPECT_EQ(
        mu,
        linalg::tensor3D({
            {1.0, 6.0, 10.0},
            {0.0, 5.0, 14.0},
            {0.0, 0.0, 9.0},
        })
    );
}

/* ---------- BerendsenManostat — type, isotropy, getters ---------- */

TEST_F(TestManostat, berendsenTauAndCompressibilityGetters)
{
    auto manostat =
        manostat::BerendsenManostat(1.0, 0.1, 4.5, settings::FixedAxis::NONE);
    EXPECT_DOUBLE_EQ(manostat.getTau(), 0.1);
    EXPECT_DOUBLE_EQ(manostat.getCompressibility(), 4.5);
}

TEST_F(TestManostat, berendsenManostatType)
{
    auto manostat =
        manostat::BerendsenManostat(1.0, 0.1, 4.5, settings::FixedAxis::NONE);
    EXPECT_EQ(manostat.getManostatType(), settings::ManostatType::BERENDSEN);
}

TEST_F(TestManostat, berendsenIsotropy)
{
    auto manostat =
        manostat::BerendsenManostat(1.0, 0.1, 4.5, settings::FixedAxis::NONE);
    EXPECT_EQ(manostat.getIsotropy(), settings::Isotropy::ISOTROPIC);
}

TEST_F(TestManostat, semiIsotropicBerendsenIsotropy)
{
    auto manostat = manostat::SemiIsotropicBerendsenManostat(
        1.0,
        0.1,
        4.5,
        2U,
        std::vector<size_t>{0U, 1U},
        settings::FixedAxis::NONE
    );
    EXPECT_EQ(manostat.getIsotropy(), settings::Isotropy::SEMI_ISOTROPIC);
    EXPECT_EQ(manostat.getManostatType(), settings::ManostatType::BERENDSEN);
}

TEST_F(TestManostat, anisotropicBerendsenIsotropy)
{
    auto manostat = manostat::AnisotropicBerendsenManostat(
        1.0,
        0.1,
        4.5,
        settings::FixedAxis::NONE
    );
    EXPECT_EQ(manostat.getIsotropy(), settings::Isotropy::ANISOTROPIC);
    EXPECT_EQ(manostat.getManostatType(), settings::ManostatType::BERENDSEN);
}

TEST_F(TestManostat, fullAnisotropicBerendsenIsotropy)
{
    auto manostat = manostat::FullAnisotropicBerendsenManostat(
        1.0,
        0.1,
        4.5,
        settings::FixedAxis::NONE
    );
    EXPECT_EQ(manostat.getIsotropy(), settings::Isotropy::FULL_ANISOTROPIC);
    EXPECT_EQ(manostat.getManostatType(), settings::ManostatType::BERENDSEN);
}

TEST_F(TestManostat, berendsenFixedAxesMu)
{
    settings::TimingsSettings::setTimeStep(0.5);

    // 1 fixed axis: X
    {
        auto manostat =
            TestableBerendsenManostat(1.0, 0.5, 0.2, settings::FixedAxis::X);
        manostat.setPressureTensor(diagonalMatrix(linalg::Vec3D(2.0, 3.0, 4.0))
        );

        const auto mu = manostat.calculateMu();
        EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
        EXPECT_DOUBLE_EQ(mu[1][1], ::sqrt(1.5));
        EXPECT_DOUBLE_EQ(mu[2][2], ::sqrt(1.5));
    }

    // 2 fixed axes: XY
    {
        auto manostat =
            TestableBerendsenManostat(1.0, 0.5, 0.2, settings::FixedAxis::XY);
        manostat.setPressureTensor(diagonalMatrix(linalg::Vec3D(2.0, 3.0, 4.0))
        );

        const auto mu = manostat.calculateMu();
        EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
        EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
        EXPECT_DOUBLE_EQ(mu[2][2], 1.6);
    }

    // All fixed axes
    {
        auto manostat =
            TestableBerendsenManostat(1.0, 0.5, 0.2, settings::FixedAxis::ALL);
        manostat.setPressureTensor(diagonalMatrix(linalg::Vec3D(2.0, 3.0, 4.0))
        );

        const auto mu = manostat.calculateMu();
        EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
        EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
        EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
    }
}

TEST_F(TestManostat, anisotropicBerendsenFixedAxesMu)
{
    settings::TimingsSettings::setTimeStep(0.5);

    auto manostat = TestableAnisotropicBerendsenManostat(
        1.0,
        0.5,
        0.2,
        settings::FixedAxis::XZ
    );
    manostat.setPressureTensor(diagonalMatrix(linalg::Vec3D(2.0, 3.0, 4.0)));

    const auto mu = manostat.calculateMu();
    EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
    EXPECT_DOUBLE_EQ(mu[1][1], 1.4);
    EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
}

TEST_F(TestManostat, fullAnisotropicBerendsenFixedAxesMu)
{
    settings::TimingsSettings::setTimeStep(0.5);

    auto manostat = TestableFullAnisotropicBerendsenManostat(
        1.0,
        0.5,
        0.2,
        settings::FixedAxis::Y
    );
    const auto pTensor =
        linalg::tensor3D({{2.0, 0.5, 0.1}, {0.5, 3.0, 0.2}, {0.1, 0.2, 4.0}});
    manostat.setPressureTensor(pTensor);

    const auto mu = manostat.calculateMu();
    // Y row and column should be zeroed except diagonal which is 1.0
    EXPECT_DOUBLE_EQ(mu[1][0], 0.0);
    EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
    EXPECT_DOUBLE_EQ(mu[1][2], 0.0);
    EXPECT_DOUBLE_EQ(mu[0][1], 0.0);
    EXPECT_DOUBLE_EQ(mu[2][1], 0.0);
}

TEST_F(TestManostat, stochasticRescalingFixedAxesMu)
{
    settings::ThermostatSettings::setActualTargetTemperature(0.0);
    settings::TimingsSettings::setTimeStep(0.5);

    // 1 fixed axis: Z
    {
        auto manostat = TestableStochasticRescalingManostat(
            7.0,
            0.25,
            0.12,
            settings::FixedAxis::Z
        );
        manostat.setPressureTensor(diagonalMatrix(linalg::Vec3D(1.0, 2.0, 3.0))
        );

        const auto mu       = manostat.calculateMu(10.0);
        const auto expected = ::exp(-(0.12 * 0.5 / 0.25) * (7.0 - 1.5) / 2.0);

        EXPECT_DOUBLE_EQ(mu[0][0], expected);
        EXPECT_DOUBLE_EQ(mu[1][1], expected);
        EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
    }

    // 2 fixed axes: XY
    {
        auto manostat = TestableStochasticRescalingManostat(
            7.0,
            0.25,
            0.12,
            settings::FixedAxis::XY
        );
        manostat.setPressureTensor(diagonalMatrix(linalg::Vec3D(1.0, 2.0, 3.0))
        );

        const auto mu       = manostat.calculateMu(10.0);
        const auto expected = ::exp(-(0.12 * 0.5 / 0.25) * (7.0 - 3.0) / 1.0);

        EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
        EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
        EXPECT_DOUBLE_EQ(mu[2][2], expected);
    }

    // All fixed axes
    {
        auto manostat = TestableStochasticRescalingManostat(
            7.0,
            0.25,
            0.12,
            settings::FixedAxis::ALL
        );
        manostat.setPressureTensor(diagonalMatrix(linalg::Vec3D(1.0, 2.0, 3.0))
        );

        const auto mu = manostat.calculateMu(10.0);

        EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
        EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
        EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
    }
}

TEST_F(TestManostat, anisotropicStochasticRescalingFixedAxesMu)
{
    settings::ThermostatSettings::setActualTargetTemperature(0.0);
    settings::TimingsSettings::setTimeStep(0.5);

    auto manostat = TestableAnisotropicStochasticRescalingManostat(
        7.0,
        0.25,
        0.12,
        settings::FixedAxis::YZ
    );
    manostat.setPressureTensor(diagonalMatrix(linalg::Vec3D(1.0, 2.0, 3.0)));

    const auto mu       = manostat.calculateMu(10.0);
    const auto expected = ::exp(-(0.12 * 0.5 / 0.25) * (7.0 - 1.0) / 3.0);

    EXPECT_DOUBLE_EQ(mu[0][0], expected);
    EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
    EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
}

TEST_F(TestManostat, fullAnisotropicStochasticRescalingFixedAxesMu)
{
    settings::ThermostatSettings::setActualTargetTemperature(0.0);
    settings::TimingsSettings::setTimeStep(0.5);

    auto manostat = TestableFullAnisotropicStochasticRescalingManostat(
        7.0,
        0.25,
        0.12,
        settings::FixedAxis::X
    );
    const auto pTensor =
        linalg::tensor3D({{1.0, 0.2, 0.3}, {0.2, 2.0, 0.4}, {0.3, 0.4, 3.0}});
    manostat.setPressureTensor(pTensor);

    const auto mu = manostat.calculateMu(10.0);
    EXPECT_DOUBLE_EQ(mu[0][0], 1.0);
    EXPECT_DOUBLE_EQ(mu[0][1], 0.0);
    EXPECT_DOUBLE_EQ(mu[0][2], 0.0);
    EXPECT_DOUBLE_EQ(mu[1][0], 0.0);
    EXPECT_DOUBLE_EQ(mu[2][0], 0.0);
}

TEST_F(TestManostat, semiIsotropicBerendsenFixedAnisotropicAxisMu)
{
    settings::TimingsSettings::setTimeStep(0.5);

    // xy isotropic (axes 0, 1), z anisotropic (axis 2) with Z fixed
    auto manostat = TestableSemiIsotropicBerendsenManostat(
        1.0,
        0.5,
        0.2,
        2U,
        std::vector<size_t>{0U, 1U},
        settings::FixedAxis::Z
    );
    manostat.setPressureTensor(diagonalMatrix(linalg::Vec3D(2.0, 4.0, 5.0)));

    const auto mu = manostat.calculateMu();
    // xy avg = 3.0, mu_xy = sqrt(1 - 0.2 * 0.5 / 0.5 * (1.0 - 3.0)) = sqrt(1.4)
    EXPECT_DOUBLE_EQ(mu[0][0], ::sqrt(1.4));
    EXPECT_DOUBLE_EQ(mu[1][1], ::sqrt(1.4));
    EXPECT_DOUBLE_EQ(mu[2][2], 1.0);
}

TEST_F(TestManostat, semiIsotropicStochasticRescalingFixedAnisotropicAxisMu)
{
    settings::ThermostatSettings::setActualTargetTemperature(0.0);
    settings::TimingsSettings::setTimeStep(0.5);

    // xz isotropic (axes 0, 2), y anisotropic (axis 1) with Y fixed
    auto manostat = TestableSemiIsotropicStochasticRescalingManostat(
        7.0,
        0.25,
        0.12,
        1U,
        std::vector<size_t>{0U, 2U},
        settings::FixedAxis::Y
    );
    manostat.setPressureTensor(diagonalMatrix(linalg::Vec3D(1.0, 5.0, 3.0)));

    const auto mu = manostat.calculateMu(10.0);
    // xz avg = 2.0, deltaPxy = 7.0 - 2.0 = 5.0
    const auto expected_xz = ::exp(-(0.12 * 0.5 / 0.25) * 5.0 / 3.0);

    EXPECT_DOUBLE_EQ(mu[0][0], expected_xz);
    EXPECT_DOUBLE_EQ(mu[1][1], 1.0);
    EXPECT_DOUBLE_EQ(mu[2][2], expected_xz);
}
