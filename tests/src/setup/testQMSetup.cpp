#include "dftbplusRunner.hpp"     // for DFTBPlusRunner
#include "exceptions.hpp"         // for InputFileException
#include "qmRunner.hpp"           // for QMRunner
#include "qmSettings.hpp"         // for QMMethod, QMSettings
#include "qmSetup.hpp"            // for QMSetup, setupQM
#include "qmmdEngine.hpp"         // for QMMDEngine
#include "throwWithMessage.hpp"   // for ASSERT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), InitGoogleTest, RUN_ALL_TESTS
#include <string>          // for allocator, basic_string

TEST(TestQMSetup, setup)
{
    engine::QMMDEngine engine;
    auto               setupQM = setup::QMSetup(engine);

    settings::QMSettings::setQMMethod(settings::QMMethod::DFTBPLUS);
    setupQM.setup();

    EXPECT_EQ(typeid(dynamic_cast<QM::DFTBPlusRunner &>(*engine.getQMRunner())), typeid(QM::DFTBPlusRunner));

    settings::QMSettings::setQMMethod(settings::QMMethod::NONE);

    ASSERT_THROW_MSG(setupQM.setup(),
                     customException::InputFileException,
                     "A qm based jobtype was requested but no external program via \"qm_prog\" provided");
}

TEST(TestQMSetup, setupQMFull)
{
    settings::QMSettings::setQMMethod(settings::QMMethod::DFTBPLUS);

    engine::QMMDEngine engine;
    EXPECT_NO_THROW(setup::setupQM(engine));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}