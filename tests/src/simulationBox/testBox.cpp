#include "testBox.hpp"

#include "constants.hpp"
#include "exceptions.hpp"
#include "vector3d.hpp"

using namespace std;
using namespace linearAlgebra;
using namespace customException;
using namespace simulationBox;
using namespace ::testing;

TEST_F(TestBox, setBoxDimensions)
{
    Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);
    EXPECT_EQ(_box->getBoxDimensions(), boxDimensions);

    boxDimensions = {-1.0, 2.0, 3.0};
    ASSERT_THROW(_box->setBoxDimensions(boxDimensions), RstFileException);
}

TEST_F(TestBox, setBoxAngles)
{
    Vec3D boxAngles = {10.0, 20.0, 30.0};
    _box->setBoxAngles(boxAngles);
    EXPECT_EQ(_box->getBoxAngles(), boxAngles);

    boxAngles = {-10.0, 20.0, 30.0};
    ASSERT_THROW(_box->setBoxAngles(boxAngles), RstFileException);

    boxAngles = {10.0, 20.0, 100.0};
    ASSERT_THROW(_box->setBoxAngles(boxAngles), RstFileException);
}

TEST_F(TestBox, setDensity)
{
    double density = 1.0;
    _box->setDensity(density);
    EXPECT_EQ(_box->getDensity(), density);

    density = -1.0;
    ASSERT_THROW(_box->setDensity(density), InputFileException);
}

TEST_F(TestBox, calculateBoxDimensionsFromDensity)
{
    const double density = 1.0;
    _box->setDensity(density / constants::_KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_);
    _box->setTotalMass(1.0);
    const Vec3D boxDimensions = {1.0, 1.0, 1.0};
    EXPECT_EQ(_box->calculateBoxDimensionsFromDensity(), boxDimensions);
}

TEST_F(TestBox, calculateVolume)
{
    const Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);
    EXPECT_EQ(_box->calculateVolume(), 6.0);
}

TEST_F(TestBox, applyPeriodiicBoundaryConditions)
{
    const Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);

    Vec3D position = {0.5, 1.5, 2.5};
    _box->applyPBC(position);

    const Vec3D expectedPosition = {0.5, -0.5, -0.5};
    EXPECT_EQ(position, expectedPosition);
}

TEST_F(TestBox, scaleBox)
{
    const Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);

    const Vec3D scaleFactors = {2.0, 2.0, 2.0};
    _box->scaleBox(scaleFactors);

    const Vec3D expectedBoxDimensions = {2.0, 4.0, 6.0};
    EXPECT_EQ(_box->getBoxDimensions(), expectedBoxDimensions);
}

TEST_F(TestBox, getMinimalBoxDimension)
{
    const Vec3D boxDimensions = {2.0, 1.0, 3.0};
    _box->setBoxDimensions(boxDimensions);

    EXPECT_EQ(_box->getMinimalBoxDimension(), 1.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}