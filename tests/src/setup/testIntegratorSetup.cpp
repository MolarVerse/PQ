#include "constants.hpp"
#include "exceptions.hpp"
#include "integratorSetup.hpp"
#include "testSetup.hpp"

using namespace setup;
using namespace customException;

TEST_F(TestSetup, integratorSetup_SetTimeStep)
{
    _engine.getTimings().setTimestep(0.001);
    IntegratorSetup integratorSetup(_engine);
    integratorSetup.setup();
    EXPECT_DOUBLE_EQ(_engine.getIntegrator().getDt(), 0.001);

    EXPECT_NO_THROW(setupIntegrator(_engine));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}