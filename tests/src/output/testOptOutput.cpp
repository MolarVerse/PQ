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

#include "convergence.hpp"
#include "convergenceSettings.hpp"
#include "optOutput.hpp"
#include "steepestDescent.hpp"

using namespace output;
using namespace opt;
using settings::ConvStrategy;

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

TEST(TestOptOutput, writeProducesStepAndAllConvergenceColumns)
{
    const std::string path = "default.opt.test";

    OptOutput out(path);
    out.setFilename(path);

    SteepestDescent opt(1u);
    Convergence     conv(
        true,
        true,
        true,
        1.0e-4,
        1.0e-4,
        1.0e-3,
        1.0e-3,
        ConvStrategy::RIGOROUS
    );
    conv.calcEnergyConvergence(1.0, 1.0 + 1.0e-5);
    conv.calcForceConvergence(1.0e-5, 2.0e-5);
    opt.setConvergence(conv);

    out.write(42, opt);
    out.close();

    const auto content = slurp(path);
    EXPECT_NE(content.find("42"), std::string::npos);
    EXPECT_NE(content.find("e-05"), std::string::npos);   // forces/energy
    EXPECT_NE(content.find("e-04"), std::string::npos);   // thresholds
    ::remove(path.c_str());
}

TEST(TestOptOutput, writeAbsoluteStrategyZeroesOutRelativeFlag)
{
    const std::string path = "default.opt.test";

    OptOutput out(path);
    out.setFilename(path);

    SteepestDescent opt(1u);
    Convergence     conv(
        true,
        true,
        true,
        1.0e-4,
        1.0e-4,
        1.0e-3,
        1.0e-3,
        ConvStrategy::ABSOLUTE
    );
    conv.calcEnergyConvergence(1.0, 1.0 + 1.0e-5);
    conv.calcForceConvergence(1.0e-5, 1.0e-5);
    opt.setConvergence(conv);

    out.write(1, opt);
    out.close();

    const auto content = slurp(path);
    // Expect a "  0" for the relative-energy convergence indicator column,
    // since ABSOLUTE strategy disables it.
    EXPECT_NE(content.find("  0\t"), std::string::npos);
    ::remove(path.c_str());
}

TEST(TestOptOutput, writeRelativeStrategyZeroesOutAbsoluteFlag)
{
    const std::string path = "default.opt.test";

    OptOutput out(path);
    out.setFilename(path);

    SteepestDescent opt(1u);
    Convergence     conv(
        true,
        true,
        true,
        1.0e-4,
        1.0e-4,
        1.0e-3,
        1.0e-3,
        ConvStrategy::RELATIVE
    );
    conv.calcEnergyConvergence(1.0, 1.0 + 1.0e-5);
    conv.calcForceConvergence(1.0e-5, 1.0e-5);
    opt.setConvergence(conv);

    out.write(1, opt);
    out.close();

    const auto content = slurp(path);
    EXPECT_NE(content.find("  0\t"), std::string::npos);
    ::remove(path.c_str());
}

TEST(TestOptOutput, writeRespectsDisabledEnergyConv)
{
    const std::string path = "default.opt.test";

    OptOutput out(path);
    out.setFilename(path);

    SteepestDescent opt(1u);
    Convergence     conv(
        false,   // energy disabled
        true,
        true,
        1.0e-4,
        1.0e-4,
        1.0e-3,
        1.0e-3,
        ConvStrategy::RIGOROUS
    );
    conv.calcForceConvergence(1.0e-5, 1.0e-5);
    opt.setConvergence(conv);

    out.write(1, opt);
    out.close();

    const auto content = slurp(path);
    // With energy disabled, both energy-conv indicator columns are "  0".
    EXPECT_NE(content.find("  0\t  0\t"), std::string::npos);
    ::remove(path.c_str());
}
