#include "constants.hpp"
#include "exceptions.hpp"
#include "potentialSetup.hpp"
#include "testSetup.hpp"

using namespace setup;

TEST_F(TestSetup, setupPotential)
{
    _engine._potential->setCoulombType("guff");
    _engine._potential->setNonCoulombType("guff");
    PotentialSetup potentialSetup(_engine);
    potentialSetup.setup();

    EXPECT_EQ(typeid(*(_engine._potential->getCoulombPotential())), typeid(potential::GuffCoulomb));
    EXPECT_EQ(typeid(*(_engine._potential->getNonCoulombPotential())), typeid(potential::GuffNonCoulomb));

    EXPECT_NO_THROW(setupPotential(_engine));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}