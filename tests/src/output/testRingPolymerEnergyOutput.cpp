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
#include "units.hpp"

class TestRingPolymerEnergyOutput : public testing::Test
{
   protected:
    static EnergyPerMol _sumOfRingPolymerEnergies(
        const std::vector<physicalData::PhysicalData> &dataVector
    );

    static EnergyPerMol _maxRingPolymerEnergy(
        const std::vector<physicalData::PhysicalData> &dataVector
    );
};

EnergyPerMol TestRingPolymerEnergyOutput::_sumOfRingPolymerEnergies(
    const std::vector<physicalData::PhysicalData> &dataVector
)
{
    return out::RingPolymerEnergyOutput::_sumOfRingPolymerEnergies(dataVector);
}

EnergyPerMol TestRingPolymerEnergyOutput::_maxRingPolymerEnergy(
    const std::vector<physicalData::PhysicalData> &dataVector
)
{
    return out::RingPolymerEnergyOutput::_maxRingPolymerEnergy(dataVector);
}

TEST_F(TestRingPolymerEnergyOutput, sumOfRingPolymerEnergiesAddsAllReplicas)
{
    out::RingPolymerEnergyOutput            out("dummy.rpe");
    std::vector<physicalData::PhysicalData> vec(3);
    vec[0].setRingPolymerEnergy(1.0 * kcal_per_mol);
    vec[1].setRingPolymerEnergy(2.5 * kcal_per_mol);
    vec[2].setRingPolymerEnergy(3.5 * kcal_per_mol);
    EXPECT_DOUBLE_EQ(
        _sumOfRingPolymerEnergies(vec).numerical_value_in(kcal_per_mol),
        7.0
    );
}

TEST_F(TestRingPolymerEnergyOutput, maxRingPolymerEnergyReturnsLargestEntry)
{
    out::RingPolymerEnergyOutput            out("dummy.rpe");
    std::vector<physicalData::PhysicalData> vec(3);
    vec[0].setRingPolymerEnergy(1.0 * kcal_per_mol);
    vec[1].setRingPolymerEnergy(9.0 * kcal_per_mol);
    vec[2].setRingPolymerEnergy(5.0 * kcal_per_mol);
    EXPECT_DOUBLE_EQ(
        _maxRingPolymerEnergy(vec).numerical_value_in(kcal_per_mol),
        9.0
    );
}

TEST_F(TestRingPolymerEnergyOutput, writeEmitsStepSumMaxMeanAndPerBeadEnergies)
{
    const std::string path = "default.rpe.test";

    out::RingPolymerEnergyOutput out(path);
    out.setFilename(path, false);

    std::vector<physicalData::PhysicalData> vec(2);
    vec[0].setRingPolymerEnergy(1.0 * kcal_per_mol);
    vec[1].setRingPolymerEnergy(3.0 * kcal_per_mol);

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
