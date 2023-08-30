#include "testBox.hpp"

#include "constants.hpp"   // for _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_
#include "vector3d.hpp"    // for Vec3D, linearAlgebra

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <memory>          // for allocator

using namespace simulationBox;

TEST_F(TestBox, setBoxDimensions)
{
    linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);
    EXPECT_EQ(_box->getBoxDimensions(), boxDimensions);
}

TEST_F(TestBox, setBoxAngles)
{
    linearAlgebra::Vec3D boxAngles = {10.0, 20.0, 30.0};
    _box->setBoxAngles(boxAngles);
    EXPECT_EQ(_box->getBoxAngles(), boxAngles);
}

TEST_F(TestBox, setDensity)
{
    double density = 1.0;
    _box->setDensity(density);
    EXPECT_EQ(_box->getDensity(), density);
}

TEST_F(TestBox, calculateBoxDimensionsFromDensity)
{
    const double density = 1.0;
    _box->setDensity(density / constants::_KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_);
    _box->setTotalMass(1.0);
    const linearAlgebra::Vec3D boxDimensions = {1.0, 1.0, 1.0};
    EXPECT_EQ(_box->calculateBoxDimensionsFromDensity(), boxDimensions);
}

TEST_F(TestBox, calculateVolume)
{
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);
    EXPECT_EQ(_box->calculateVolume(), 6.0);
}

TEST_F(TestBox, applyPeriodicBoundaryConditions)
{
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);

    linearAlgebra::Vec3D position = {0.5, 1.5, 2.5};
    _box->applyPBC(position);

    const linearAlgebra::Vec3D expectedPosition = {0.5, -0.5, -0.5};
    EXPECT_EQ(position, expectedPosition);
}

TEST_F(TestBox, scaleBox)
{
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);

    const linearAlgebra::Vec3D scaleFactors = {2.0, 2.0, 2.0};
    _box->scaleBox(scaleFactors);

    const linearAlgebra::Vec3D expectedBoxDimensions = {2.0, 4.0, 6.0};
    EXPECT_EQ(_box->getBoxDimensions(), expectedBoxDimensions);
}

TEST_F(TestBox, getMinimalBoxDimension)
{
    const linearAlgebra::Vec3D boxDimensions = {2.0, 1.0, 3.0};
    _box->setBoxDimensions(boxDimensions);

    EXPECT_EQ(_box->getMinimalBoxDimension(), 1.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}