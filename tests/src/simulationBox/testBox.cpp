#include "testBox.hpp"

#include "constants.hpp"
#include "exceptions.hpp"
#include "vector3d.hpp"

using namespace std;
using namespace vector3d;
using namespace customException;
using namespace simulationBox;
using namespace config;
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
    double density = 1.0;
    _box->setDensity(density / _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_);
    _box->setTotalMass(1.0);
    Vec3D boxDimensions = {1.0, 1.0, 1.0};
    EXPECT_EQ(_box->calculateBoxDimensionsFromDensity(), boxDimensions);
}

TEST_F(TestBox, calculateVolume)
{
    Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);
    EXPECT_EQ(_box->calculateVolume(), 6.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}