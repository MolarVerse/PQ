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

#include "physicalData.hpp"
#include "vector3d.hpp"   // IWYU pragma: keep
#include "virialOutput.hpp"

using namespace output;
using physicalData::PhysicalData;

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

TEST(TestVirialOutput, writeEmitsStepAndAllNineTensorComponents)
{
    const std::string path = "default.vir.test";

    VirialOutput out(path);
    out.setFilename(path);

    PhysicalData data;
    data.setVirial(
        linearAlgebra::tensor3D{
            linearAlgebra::Vec3D{0.1, 0.2, 0.3},
            linearAlgebra::Vec3D{0.4, 0.5, 0.6},
            linearAlgebra::Vec3D{0.7, 0.8, 0.9}
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
    for (auto c : content)
        if (c == '\n')
            ++newlines;
    EXPECT_EQ(newlines, 3u);

    const auto errorCode = std::remove(path.c_str());
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: " << path;
}
