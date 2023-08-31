#include "testManostat.hpp"

#include "constants.hpp"          // for _PRESSURE_FACTOR_
#include "exceptions.hpp"         // for ManostatException
#include "mathUtilities.hpp"      // for compare
#include "molecule.hpp"           // for Molecule
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG
#include "vector3d.hpp"           // for Vector3D, Vec3D

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <cmath>           // for pow
#include <string>          // for string, allocator

/**
 * @brief tests function calculate pressure
 *
 */
TEST_F(TestManostat, CalculatePressure)
{
    _manostat->calculatePressure(*_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * constants::_PRESSURE_FACTOR_);
}

/**
 * @brief tests function to change virial to atomic
 *
 */
TEST_F(TestManostat, ChangeVirialToAtomic)
{
    _data->changeKineticVirialToAtomic();

    _manostat->calculatePressure(*_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 2.0 * constants::_PRESSURE_FACTOR_);
}

/**
 * @brief tests application of berendsen manostat
 *
 */
TEST_F(TestManostat, testApplyBerendsenManostat)
{
    _box->setCoulombRadiusCutOff(0.99);
    _box->setBoxDimensions({2.0, 2.0, 2.0});
    auto boxOld = _box->getBoxDimensions();

    auto molecule = simulationBox::Molecule();
    molecule.addAtomPosition({1.0, 0.0, 0.0});
    molecule.setCenterOfMass({1.0, 0.0, 0.0});
    molecule.setNumberOfAtoms(1);

    _box->addMolecule(molecule);

    _manostat = new manostat::BerendsenManostat(1.0, 0.1, 4.5);
    _manostat->setTimestep(0.5);

    const auto scaleFactors =
        linearAlgebra::Vec3D(::pow(1.0 - 4.5 * 0.5 / 0.1 * (1.0 - 3.0 * constants::_PRESSURE_FACTOR_), 1.0 / 3.0));

    _manostat->applyManostat(*_box, *_data);
    auto boxNew = _box->getBoxDimensions();

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * constants::_PRESSURE_FACTOR_);
    EXPECT_DOUBLE_EQ(boxNew[0], (boxOld * scaleFactors)[0]);
    EXPECT_DOUBLE_EQ(boxNew[1], (boxOld * scaleFactors)[1]);
    EXPECT_DOUBLE_EQ(boxNew[2], (boxOld * scaleFactors)[2]);
    EXPECT_TRUE(
        utilities::compare(_box->getMolecule(0).getAtomPosition(0), linearAlgebra::Vec3D(1.0, 0.0, 0.0) * scaleFactors, 1e-9));
}

/**
 * @brief tests application of berendsen manostat if coulomb radius is larger than half of the minimum box dimension
 *
 */
TEST_F(TestManostat, testApplyBerendsenManostat_cutoffLargerThanHalfOfMinimumBoxDimension)
{
    _box->setCoulombRadiusCutOff(10.0);
    _box->setBoxDimensions({2.0, 2.0, 2.0});

    _manostat = new manostat::BerendsenManostat(3.0 * constants::_PRESSURE_FACTOR_, 0.1, 4.5);
    _manostat->setTimestep(0.5);

    EXPECT_THROW_MSG(_manostat->applyManostat(*_box, *_data),
                     customException::ManostatException,
                     "Coulomb radius cut off is larger than half of the minimal box dimension");
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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}