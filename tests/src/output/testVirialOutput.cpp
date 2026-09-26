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

#include <gtest/gtest.h>

#include <cstdio>
#include <string>

#include "physicalData.hpp"
#include "testOutputBase.hpp"
#include "vector3d.hpp"
#include "virialOutput.hpp"

using namespace out;
using physicalData::PhysicalData;

TEST(TestVirialOutput, writeEmitsStepAndAllNineTensorComponents)
{
    const std::string path = "default.vir.test";

    VirialOutput out(path);
    out.setFilename(path);

    PhysicalData data;
    data.setVirial(
        linalg::tensor3D{
            linalg::Vec3D{0.1, 0.2, 0.3},
            linalg::Vec3D{0.4, 0.5, 0.6},
            linalg::Vec3D{0.7, 0.8, 0.9}
        }
    );

    out.write(42, data);
    out.close();

    const auto content = slurp(path);
    EXPECT_NE(content.find("42"), std::string::npos);
    EXPECT_NE(content.find("1.00000e-01"), std::string::npos);
    EXPECT_NE(content.find("5.00000e-01"), std::string::npos);
    EXPECT_NE(content.find("9.00000e-01"), std::string::npos);

    const auto errorCode = std::remove(path.c_str());
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: " << path;
}

TEST(TestVirialOutput, writeEmitsOneLinePerCall)
{
    const std::string path = "default.vir.test";

    VirialOutput out(path);
    out.setFilename(path);

    PhysicalData data;

    out.write(1, data);
    out.write(2, data);
    out.write(3, data);
    out.close();

    const auto content  = slurp(path);
    size_t     newlines = 0;
    for (auto character : content)
        if (character == '\n')
            ++newlines;
    EXPECT_EQ(newlines, 3U);

    const auto errorCode = std::remove(path.c_str());
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: " << path;
}
