#include "constants.hpp"
#include "exceptions.hpp"
#include "manostatSetup.hpp"
#include "testSetup.hpp"

using namespace setup;
using namespace customException;

TEST_F(TestSetup, setup)
{
    ManostatSetup manostatSetup(_engine);

    _engine.getTimings().setTimestep(0.1);
    manostatSetup.setup();

    EXPECT_EQ(_engine.getManostat().getTimestep(), 0.1);

    _engine.getSettings().setManostat("berendsen");
    EXPECT_THROW(manostatSetup.setup(), InputFileException);

    _engine.getSettings().setPressure(300.0);
    EXPECT_NO_THROW(manostatSetup.setup());

    auto berendsenManostat = dynamic_cast<manostat::BerendsenManostat &>(_engine.getManostat());
    EXPECT_EQ(berendsenManostat.getTau(), 1.0 * 1000);

    _engine.getSettings().setTauManostat(0.2);
    EXPECT_NO_THROW(manostatSetup.setup());

    auto berendsenManostat2 = dynamic_cast<manostat::BerendsenManostat &>(_engine.getManostat());
    EXPECT_EQ(berendsenManostat2.getTau(), 0.2 * 1000);

    EXPECT_NO_THROW(setupManostat(_engine));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}