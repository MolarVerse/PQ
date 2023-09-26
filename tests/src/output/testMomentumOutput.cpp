#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "momentumOutput.hpp"       // for MomentumOutput
#include "physicalData.hpp"         // for PhysicalData
#include "settings.hpp"             // for Settings
#include "testEnergyOutput.hpp"     // for TestEnergyOutput
#include "vector3d.hpp"             // for Vec3D

#include "gtest/gtest.h"   // for Test, TestInfo (ptr only), TEST, InitGoogleTest, RUN_ALL_TESTS
#include <fstream>         // for ifstream
#include <gtest/gtest.h>   // for Message, TestPartResult
#include <string>          // for getline, allocator, string

/**
 * @brief tests writing momentum output file
 *
 */
TEST_F(TestEnergyOutput, writeMomentumFile)
{
    _physicalData->setMomentum(linearAlgebra::Vec3D(3.0, 4.0, 0.0));
    _physicalData->setAngularMomentum(linearAlgebra::Vec3D(4.0, 0.0, 3.0));

    settings::ForceFieldSettings::deactivate();
    settings::Settings::activateMM();

    _momentumOutput->setFilename("default.mom");
    _momentumOutput->write(100, *_physicalData);
    _momentumOutput->close();

    std::ifstream file("default.mom");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line,
              "       100\t         5.00000e+00\t         3.00000e+00\t         4.00000e+00\t         0.00000e+00\t         "
              "5.00000e+00\t         4.00000e+00\t         0.00000e+00\t         3.00000e+00");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}