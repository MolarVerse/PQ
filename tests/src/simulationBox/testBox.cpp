#include "testBox.hpp"
#include "exceptions.hpp"
#include "constants.hpp"

using namespace std;
using namespace ::testing;

TEST_F(TestBox, setBoxDimensions)
{
    vector<double> boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);
    EXPECT_EQ(_box->getBoxDimensions(), boxDimensions);
}

TEST_F(TestBox, setBoxDimensionsNegative)
{
    vector<double> boxDimensions = {-1.0, 2.0, 3.0};
    ASSERT_THROW(_box->setBoxDimensions(boxDimensions), RstFileException);
}

TEST_F(TestBox, setBoxAngles)
{
    vector<double> boxAngles = {10.0, 20.0, 30.0};
    _box->setBoxAngles(boxAngles);
    EXPECT_EQ(_box->getBoxAngles(), boxAngles);
}

TEST_F(TestBox, setBoxAnglesNegative)
{
    vector<double> boxAngles = {-10.0, 20.0, 30.0};
    ASSERT_THROW(_box->setBoxAngles(boxAngles), RstFileException);
}

TEST_F(TestBox, setBoxAnglesGreaterThan90)
{
    vector<double> boxAngles = {10.0, 20.0, 100.0};
    ASSERT_THROW(_box->setBoxAngles(boxAngles), RstFileException);
}

TEST_F(TestBox, setDensity)
{
    double density = 1.0;
    _box->setDensity(density);
    EXPECT_EQ(_box->getDensity(), density);
}

TEST_F(TestBox, setDensityNegative)
{
    double density = -1.0;
    ASSERT_THROW(_box->setDensity(density), InputFileException);
}

TEST_F(TestBox, calculateBoxDimensionsFromDensity)
{
    double density = 1.0;
    _box->setDensity(density / _KG_PER_LITER_CUBIC_TO_AMU_PER_ANGSTROM_CUBIC_);
    _box->setTotalMass(1.0);
    vector<double> boxDimensions = {1.0, 1.0, 1.0};
    EXPECT_EQ(_box->calculateBoxDimensionsFromDensity(), boxDimensions);
}

TEST_F(TestBox, calculateVolume)
{
    vector<double> boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);
    EXPECT_EQ(_box->calculateVolume(), 6.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}