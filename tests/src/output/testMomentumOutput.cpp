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

#include <gtest/gtest.h>   // for Message, TestPartResult

#include <fstream>   // for ifstream
#include <string>    // for getline, allocator, string

#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "gtest/gtest.h"   // for Test, TestInfo (ptr only), TEST, InitGoogleTest, RUN_ALL_TESTS
#include "momentumOutput.hpp"     // for MomentumOutput
#include "physicalData.hpp"       // for PhysicalData
#include "settings.hpp"           // for Settings
#include "testEnergyOutput.hpp"   // for TestEnergyOutput
#include "vector3d.hpp"           // for Vec3D

using namespace linearAlgebra;

/**
 * @brief tests writing momentum output file
 *
 */
TEST_F(TestEnergyOutput, writeMomentumFile)
{
    _physicalData->setMomentum(Vec3D(3.0, 4.0, 0.0));
    _physicalData->setAngularMomentum(Vec3D(4.0, 0.0, 3.0));

    _momentumOutput->setFilename("default.mom");
    _momentumOutput->write(100, *_physicalData);
    _momentumOutput->close();

    std::ifstream file("default.mom");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "       100\t         5.00000e+00\t         3.00000e+00\t         "
        "4.00000e+00\t         0.00000e+00\t         "
        "5.00000e+00\t         4.00000e+00\t         0.00000e+00\t         "
        "3.00000e+00"
    );
}