// #include "engine.hpp"                  // for Engine
// #include "resetKinetics.hpp"           // for ResetMomentum, ResetTemperature
// #include "resetKineticsSettings.hpp"   // for ResetKineticsSettings
// #include "resetKineticsSetup.hpp"      // for ResetKineticsSetup, setupResetKine...
// #include "testSetup.hpp"               // for TestSetup
// #include "timingsSettings.hpp"         // for TimingsSettings

// #include "gtest/gtest.h"   // for Message, TestPartResult
// #include <gtest/gtest.h>   // for EXPECT_EQ, InitGoogleTest, RUN_ALL...
// #include <string>          // for allocator, basic_string

// using namespace setup;

// TEST_F(TestSetup, setup)
// {
//     ResetKineticsSetup resetKineticsSetup(*_engine);

//     settings::TimingsSettings::setNumberOfSteps(100);

//     resetKineticsSetup.setup();
//     const auto resetKinetics = dynamic_cast<resetKinetics::ResetKinetics &>(_engine->getResetKinetics());
//     EXPECT_EQ(typeid(resetKinetics), typeid(resetKinetics::ResetKinetics));

//     settings::ResetKineticsSettings::setNScale(1);
//     resetKineticsSetup.setup();
//     const auto resetKinetics2 = dynamic_cast<resetKinetics::ResetTemperature &>(_engine->getResetKinetics());
//     EXPECT_EQ(typeid(resetKinetics2), typeid(resetKinetics::ResetTemperature));
//     EXPECT_EQ(resetKinetics2.getNStepsTemperatureReset(), 1);
//     EXPECT_EQ(resetKinetics2.getFrequencyTemperatureReset(), 100 + 1);
//     EXPECT_EQ(resetKinetics2.getNStepsMomentumReset(), 0);
//     EXPECT_EQ(resetKinetics2.getFrequencyMomentumReset(), 100 + 1);

//     settings::ResetKineticsSettings::setNScale(0);
//     settings::ResetKineticsSettings::setFScale(1);
//     resetKineticsSetup.setup();
//     const auto resetKinetics3 = dynamic_cast<resetKinetics::ResetTemperature &>(_engine->getResetKinetics());
//     EXPECT_EQ(typeid(resetKinetics3), typeid(resetKinetics::ResetTemperature));
//     EXPECT_EQ(resetKinetics3.getNStepsTemperatureReset(), 0);
//     EXPECT_EQ(resetKinetics3.getFrequencyTemperatureReset(), 1);
//     EXPECT_EQ(resetKinetics3.getNStepsMomentumReset(), 0);
//     EXPECT_EQ(resetKinetics3.getFrequencyMomentumReset(), 100 + 1);

//     settings::ResetKineticsSettings::setFScale(0);
//     settings::ResetKineticsSettings::setNReset(1);
//     resetKineticsSetup.setup();
//     const auto resetKinetics4 = dynamic_cast<resetKinetics::ResetMomentum &>(_engine->getResetKinetics());
//     EXPECT_EQ(typeid(resetKinetics4), typeid(resetKinetics::ResetMomentum));
//     EXPECT_EQ(resetKinetics4.getNStepsTemperatureReset(), 0);
//     EXPECT_EQ(resetKinetics4.getFrequencyTemperatureReset(), 100 + 1);
//     EXPECT_EQ(resetKinetics4.getNStepsMomentumReset(), 1);
//     EXPECT_EQ(resetKinetics4.getFrequencyMomentumReset(), 100 + 1);

//     settings::ResetKineticsSettings::setNReset(0);
//     settings::ResetKineticsSettings::setFReset(1);
//     resetKineticsSetup.setup();
//     const auto resetKinetics5 = dynamic_cast<resetKinetics::ResetMomentum &>(_engine->getResetKinetics());
//     EXPECT_EQ(typeid(resetKinetics5), typeid(resetKinetics::ResetMomentum));
//     EXPECT_EQ(resetKinetics5.getNStepsTemperatureReset(), 0);
//     EXPECT_EQ(resetKinetics5.getFrequencyTemperatureReset(), 100 + 1);
//     EXPECT_EQ(resetKinetics5.getNStepsMomentumReset(), 0);
//     EXPECT_EQ(resetKinetics5.getFrequencyMomentumReset(), 1);

//     EXPECT_NO_THROW(setupResetKinetics(*_engine));
// }

// int main(int argc, char **argv)
// {
//     ::testing::InitGoogleTest(&argc, argv);
//     return ::RUN_ALL_TESTS();
// }