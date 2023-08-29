#include "engine.hpp"             // for Engine
#include "exceptions.hpp"         // for InputFileException, customException
#include "manostat.hpp"           // for BerendsenManostat, Manostat
#include "manostatSettings.hpp"   // for ManostatSettings
#include "manostatSetup.hpp"      // for ManostatSetup, setupManostat, setup
#include "testSetup.hpp"          // for TestSetup
#include "timingsSettings.hpp"    // for TimingsSettings

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, EXPECT_NO_THROW, InitGoog...
#include <string>          // for allocator, basic_string

using namespace setup;
using namespace customException;

/**
 * @TODO: refactor this test to use the new setupManostat function
 *
 * @TODO: include compressibility in the test
 *
 */
TEST_F(TestSetup, setup)
{
    ManostatSetup manostatSetup(_engine);

    settings::TimingsSettings::setTimeStep(0.1);
    manostatSetup.setup();

    EXPECT_EQ(_engine.getManostat().getTimestep(), 0.1);

    settings::ManostatSettings::setManostatType("berendsen");
    EXPECT_THROW(manostatSetup.setup(), InputFileException);

    settings::ManostatSettings::setPressureSet(true);
    settings::ManostatSettings::setTargetPressure(300.0);
    EXPECT_NO_THROW(manostatSetup.setup());

    auto berendsenManostat = dynamic_cast<manostat::BerendsenManostat &>(_engine.getManostat());
    EXPECT_EQ(berendsenManostat.getTau(), 1.0 * 1000);

    settings::ManostatSettings::setTauManostat(0.2);
    EXPECT_NO_THROW(manostatSetup.setup());

    auto berendsenManostat2 = dynamic_cast<manostat::BerendsenManostat &>(_engine.getManostat());
    EXPECT_EQ(berendsenManostat2.getTau(), 0.2 * 1000);

    EXPECT_NO_THROW(setupManostat(_engine));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}