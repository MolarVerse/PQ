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
#include <thread>

#include "globalTimer.hpp"
#include "timer.hpp"
#include "timingsOutput.hpp"

using namespace output;
using namespace timings;

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

TEST(TestTimingsOutput, writeProducesHeaderAndTotalRow)
{
    const std::string path = "default.timings.test";

    TimingsOutput out(path);
    out.setFilename(path);

    GlobalTimer global;
    global.startSimulationTimer();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    global.stopSimulationTimer();

    out.write(global);
    out.close();

    const auto content = slurp(path);
    EXPECT_NE(content.find("Section"), std::string::npos);
    EXPECT_NE(content.find("Time [s]"), std::string::npos);
    EXPECT_NE(content.find("Time [%]"), std::string::npos);
    EXPECT_NE(content.find("Total"), std::string::npos);
    EXPECT_NE(content.find("RelT [%]"), std::string::npos);
    const auto errorCode = std::remove(path.c_str());
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: " << path;
}

TEST(TestTimingsOutput, writeListsRegisteredSubTimers)
{
    const std::string path = "default.timings.test";

    TimingsOutput out(path);
    out.setFilename(path);

    GlobalTimer global;
    global.startSimulationTimer();

    Timer t("MySection");
    t.startTimingsSection("inner");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    t.stopTimingsSection("inner");

    global.addTimer(t);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    global.stopSimulationTimer();

    out.write(global);
    out.close();

    const auto content = slurp(path);
    EXPECT_NE(content.find("MySection"), std::string::npos);
    EXPECT_NE(content.find("inner"), std::string::npos);
    const auto errorCode = std::remove(path.c_str());
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: " << path;
}
