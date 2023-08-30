#include "engine.hpp"            // for Engine
#include "integrator.hpp"        // for Integrator
#include "integratorSetup.hpp"   // for setupIntegrator, IntegratorSetup, setup
#include "testSetup.hpp"         // for TestSetup
#include "timingsSettings.hpp"   // for TimingsSettings

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for CmpHelperFloatingPointEQ, InitGoogleTest
#include <string>          // for allocator, basic_string

using namespace setup;

TEST_F(TestSetup, integratorSetup_SetTimeStep)
{
    settings::TimingsSettings::setTimeStep(0.001);
    IntegratorSetup integratorSetup(_engine);
    integratorSetup.setup();
    EXPECT_DOUBLE_EQ(_engine.getIntegrator().getDt(), 0.001);

    EXPECT_NO_THROW(setupIntegrator(_engine));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}