// #include "constants.hpp"
// #include "exceptions.hpp"
// #include "potentialSetup.hpp"
// #include "testSetup.hpp"

// using namespace setup;

// TEST_F(TestSetup, setupCoulombPotential)
// {
//     _engine.getPotential().setCoulombType("guff");
//     PotentialSetup potentialSetup(_engine);
//     potentialSetup.setup();

//     EXPECT_EQ(typeid(*(_engine.getPotential().getCoulombPotential())), typeid(potential::GuffCoulomb));
//     EXPECT_NO_THROW(setupPotential(_engine));

//     _engine.getSettings().setCoulombLongRangeType("wolf");
//     PotentialSetup potentialSetup2(_engine);
//     potentialSetup2.setup();

//     EXPECT_EQ(typeid(*(_engine.getPotential().getCoulombPotential())), typeid(potential::GuffWolfCoulomb));
//     const auto *wolfCoulomb = dynamic_cast<potential::GuffWolfCoulomb *>(_engine.getPotential().getCoulombPotential());
//     EXPECT_EQ(wolfCoulomb->getKappa(), 0.25);
// }

// TEST_F(TestSetup, setupNonCoulombPotential)
// {
//     _engine.getPotential().setNonCoulombType("guff");
//     PotentialSetup potentialSetup(_engine);
//     potentialSetup.setup();

//     EXPECT_EQ(typeid(*(_engine.getPotential().getNonCoulombPotential())), typeid(potential::GuffNonCoulomb));

//     EXPECT_NO_THROW(setupPotential(_engine));

//     _engine.getSettings().setNonCoulombType("lj");
//     PotentialSetup potentialSetup2(_engine);
//     potentialSetup2.setup();
//     EXPECT_EQ(typeid(*(_engine.getPotential().getNonCoulombPotential())), typeid(potential::GuffLennardJones));

//     _engine.getSettings().setNonCoulombType("buck");
//     PotentialSetup potentialSetup3(_engine);
//     potentialSetup3.setup();
//     EXPECT_EQ(typeid(*(_engine.getPotential().getNonCoulombPotential())), typeid(potential::GuffBuckingham));
// }

// int main(int argc, char **argv)
// {
//     ::testing::InitGoogleTest(&argc, argv);
//     return ::RUN_ALL_TESTS();
// }