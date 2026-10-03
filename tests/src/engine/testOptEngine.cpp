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

#include <string>
#include <vector>

#include "exceptions.hpp"
#include "optEngine.hpp"
#include "throwWithMessage.hpp"

TEST(TestOptEngine, aConvergedRunDoesNotThrow)
{
    EXPECT_NO_THROW(engine::OptEngine::throwOnFailure(true, false, 4, 10, {}));
}

TEST(TestOptEngine, aRunWithoutConvergenceThrows)
{
    EXPECT_THROW_MSG(
        engine::OptEngine::throwOnFailure(false, false, 10, 10, {}),
        exc::OptException,
        "Optimizer did not converge after 10 epochs."
    );
}

TEST(TestOptEngine, aStoppedRunReportsTheLearningRateErrors)
{
    // the engine only sets "stopped" for a run that has not converged
    EXPECT_THROW_MSG(
        engine::OptEngine::throwOnFailure(
            false,
            true,
            4,
            10,
            std::vector<std::string>{"learning rate too small", "stalled"}
        ),
        exc::OptException,
        "Optimizer stopped after 4 epochs out of 10. The following error "
        "messages were raised:\n1) learning rate too small\n2) stalled\n"
    );
}

TEST(TestOptEngine, aRunThatHitsTheEpochLimitIsNotReportedAsStopped)
{
    try
    {
        engine::OptEngine::throwOnFailure(false, false, 10, 10, {"unused"});
        FAIL() << "expected an OptException";
    }
    catch (const exc::OptException &exception)
    {
        EXPECT_EQ(
            std::string(exception.what()).find("stopped"),
            std::string::npos
        );
    }
}
