#include "testSimulationBox.hpp"

TEST_F(TestSimulationBox, resizeGuff)
{
    _simulationBox->resizeGuff(2);
    EXPECT_EQ(_simulationBox->getGuffCoefficients().size(), 2);
    EXPECT_EQ(_simulationBox->getRncCutOffs().size(), 2);
    EXPECT_EQ(_simulationBox->getCoulombCoefficients().size(), 2);
    EXPECT_EQ(_simulationBox->getcEnergyCutOffs().size(), 2);
    EXPECT_EQ(_simulationBox->getcForceCutOffs().size(), 2);

    _simulationBox->resizeGuff(0, 2);
    EXPECT_EQ(_simulationBox->getGuffCoefficients()[0].size(), 2);
    EXPECT_EQ(_simulationBox->getRncCutOffs()[0].size(), 2);
    EXPECT_EQ(_simulationBox->getCoulombCoefficients()[0].size(), 2);
    EXPECT_EQ(_simulationBox->getcEnergyCutOffs()[0].size(), 2);
    EXPECT_EQ(_simulationBox->getcForceCutOffs()[0].size(), 2);

    _simulationBox->resizeGuff(0, 0, 2);
    EXPECT_EQ(_simulationBox->getGuffCoefficients()[0][0].size(), 2);
    EXPECT_EQ(_simulationBox->getRncCutOffs()[0][0].size(), 2);
    EXPECT_EQ(_simulationBox->getCoulombCoefficients()[0][0].size(), 2);
    EXPECT_EQ(_simulationBox->getcEnergyCutOffs()[0][0].size(), 2);
    EXPECT_EQ(_simulationBox->getcForceCutOffs()[0][0].size(), 2);

    _simulationBox->resizeGuff(0, 0, 0, 2);
    EXPECT_EQ(_simulationBox->getGuffCoefficients()[0][0][0].size(), 2);
    EXPECT_EQ(_simulationBox->getRncCutOffs()[0][0][0].size(), 2);
    EXPECT_EQ(_simulationBox->getCoulombCoefficients()[0][0][0].size(), 2);
    EXPECT_EQ(_simulationBox->getcEnergyCutOffs()[0][0][0].size(), 2);
    EXPECT_EQ(_simulationBox->getcForceCutOffs()[0][0][0].size(), 2);
}

TEST_F(TestSimulationBox, numberOfAtoms) { EXPECT_EQ(_simulationBox->getNumberOfAtoms(), 5); }

TEST_F(TestSimulationBox, calculateDegreesOfFreedom)
{
    _simulationBox->calculateDegreesOfFreedom();
    EXPECT_EQ(_simulationBox->getDegreesOfFreedom(), 15);
}

TEST_F(TestSimulationBox, centerOfMassOfMolecules)
{
    _simulationBox->calculateCenterOfMassMolecules();

    auto molecules = _simulationBox->getMolecules();

    EXPECT_EQ(molecules[0].getCenterOfMass(), vector3d::Vec3D(1 / 3.0, 0.5, 0.0));
    EXPECT_EQ(molecules[1].getCenterOfMass(), vector3d::Vec3D(2 / 3.0, 0.0, 0.0));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
