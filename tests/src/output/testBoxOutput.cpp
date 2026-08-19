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
#include <fstream>
#include <sstream>
#include <string>

#include "boxOutput.hpp"
#include "orthorhombicBox.hpp"
#include "vector3d.hpp"   // IWYU pragma: keep

using namespace output;
using simulationBox::OrthorhombicBox;

namespace
{
    std::string slurp(const std::string &path)
    {
        std::ifstream     in(path);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
}   // namespace

TEST(TestBoxFileOutput, writeEmitsStepAndDimensionsAndAngles)
{
    const std::string path = "default.box.test";

    BoxFileOutput out(path);
    out.setFilename(path);

    OrthorhombicBox box;
    box.setBoxDimensions(linearAlgebra::Vec3D(10.0, 20.0, 30.0));

    out.write(7, box);
    out.close();

    const auto content = slurp(path);
    EXPECT_NE(content.find("7"), std::string::npos);
    EXPECT_NE(content.find("10.00000000"), std::string::npos);
    EXPECT_NE(content.find("20.00000000"), std::string::npos);
    EXPECT_NE(content.find("30.00000000"), std::string::npos);
    // Orthorhombic angles: 90 / 90 / 90.
    EXPECT_NE(content.find("90.00000000"), std::string::npos);

    const auto errorCode = std::remove(path.c_str());
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: " << path;
}

TEST(TestBoxFileOutput, writeOneLinePerCall)
{
    const std::string path = "default.box.test";

    BoxFileOutput out(path);
    out.setFilename(path);

    OrthorhombicBox box;
    box.setBoxDimensions(linearAlgebra::Vec3D(1.0, 2.0, 3.0));

    out.write(1, box);
    out.write(2, box);
    out.close();

    const auto content = slurp(path);
    // Two newlines for two steps written.
    size_t newlines = 0;
    for (auto c : content)
        if (c == '\n')
            ++newlines;
    EXPECT_EQ(newlines, 2U);

    const auto errorCode = std::remove(path.c_str());
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: " << path;
}
