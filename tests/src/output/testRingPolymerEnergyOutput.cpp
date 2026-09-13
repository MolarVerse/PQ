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
#include <vector>

#include "physicalData.hpp"
#include "ringPolymerEnergyOutput.hpp"
#include "testOutputBase.hpp"

using namespace out;
using physicalData::PhysicalData;

class TestRingPolymerEnergyOutput : public testing::Test
{
   protected:
    static double sumOfRingPolymerEnergies(
        const std::vector<PhysicalData> &dataVector
    )
    {
        return RingPolymerEnergyOutput::_sumOfRingPolymerEnergies(dataVector);
    }

    static double maxRingPolymerEnergy(
        const std::vector<PhysicalData> &dataVector
    )
    {
        return RingPolymerEnergyOutput::_maxRingPolymerEnergy(dataVector);
    }
};

TEST_F(TestRingPolymerEnergyOutput, sumOfRingPolymerEnergiesAddsAllReplicas)
{
    RingPolymerEnergyOutput   out("dummy.rpe");
    std::vector<PhysicalData> vec(3);
    vec[0].setRingPolymerEnergy(1.0);
    vec[1].setRingPolymerEnergy(2.5);
    vec[2].setRingPolymerEnergy(3.5);
    EXPECT_DOUBLE_EQ(sumOfRingPolymerEnergies(vec), 7.0);
}

TEST_F(TestRingPolymerEnergyOutput, maxRingPolymerEnergyReturnsLargestEntry)
{
    RingPolymerEnergyOutput   out("dummy.rpe");
    std::vector<PhysicalData> vec(3);
    vec[0].setRingPolymerEnergy(1.0);
    vec[1].setRingPolymerEnergy(9.0);
    vec[2].setRingPolymerEnergy(5.0);
    EXPECT_DOUBLE_EQ(maxRingPolymerEnergy(vec), 9.0);
}

TEST_F(TestRingPolymerEnergyOutput, writeEmitsStepSumMaxMeanAndPerBeadEnergies)
{
    const std::string path = "default.rpe.test";

    RingPolymerEnergyOutput out(path);
    out.setFilename(path);

    std::vector<PhysicalData> vec(2);
    vec[0].setRingPolymerEnergy(1.0);
    vec[1].setRingPolymerEnergy(3.0);

    out.write(5, vec);
    out.close();

    const auto content = slurp(path);
    EXPECT_NE(content.find('5'), std::string::npos);
    // Sum = 4.0
    EXPECT_NE(content.find("4.000000000000"), std::string::npos);
    // Max = 3.0 (also serves as the second value)
    EXPECT_NE(content.find("3.000000000000"), std::string::npos);
    // Mean = sum / 2 = 2.0
    EXPECT_NE(content.find("2.000000000000"), std::string::npos);
    // Individual entries (1.0 and 3.0)
    EXPECT_NE(content.find("1.000000000000"), std::string::npos);

    const auto errorCode = std::remove(path.c_str());
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: " << path;
}
