#include "testResetKinetics.hpp"

TEST_F(TestResetKinetics, resetTemperature)
{
    const auto velocity_mol1_atom1_old = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_old = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_old = _simulationBox->getMolecule(1).getAtomVelocity(0);

    _data->setTemperature(100.0);
    _resetKinetics->resetTemperature(*_data, *_simulationBox);

    const auto velocity_mol1_atom1_new = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_DOUBLE_EQ(velocity_mol1_atom1_new[0] / velocity_mol1_atom1_old[0], sqrt(3.0));
    EXPECT_DOUBLE_EQ(velocity_mol1_atom1_new[1] / velocity_mol1_atom1_old[1], sqrt(3.0));
    EXPECT_DOUBLE_EQ(velocity_mol1_atom1_new[2] / velocity_mol1_atom1_old[2], sqrt(3.0));
    EXPECT_DOUBLE_EQ(velocity_mol1_atom2_new[0] / velocity_mol1_atom2_old[0], sqrt(3.0));
    EXPECT_DOUBLE_EQ(velocity_mol1_atom2_new[1] / velocity_mol1_atom2_old[1], sqrt(3.0));
    EXPECT_DOUBLE_EQ(velocity_mol1_atom2_new[2] / velocity_mol1_atom2_old[2], sqrt(3.0));
    EXPECT_DOUBLE_EQ(velocity_mol2_atom1_new[0] / velocity_mol2_atom1_old[0], sqrt(3.0));
    EXPECT_DOUBLE_EQ(velocity_mol2_atom1_new[1] / velocity_mol2_atom1_old[1], sqrt(3.0));
    EXPECT_DOUBLE_EQ(velocity_mol2_atom1_new[2] / velocity_mol2_atom1_old[2], sqrt(3.0));
}

TEST_F(TestResetKinetics, resetMomentum)
{
    _data->setMomentumVector(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
    _resetKinetics->resetMomentum(*_data, *_simulationBox);

    const auto velocity_mol1_atom1 = _simulationBox->getMolecule(0).getAtomVelocity(0) - linearAlgebra::Vec3D(1.0, 2.0, 3.0) / 3.0;
    const auto velocity_mol1_atom2 = _simulationBox->getMolecule(0).getAtomVelocity(1) - linearAlgebra::Vec3D(1.0, 2.0, 3.0) / 3.0;
    const auto velocity_mol2_atom1 = _simulationBox->getMolecule(1).getAtomVelocity(0) - linearAlgebra::Vec3D(1.0, 2.0, 3.0) / 3.0;

    EXPECT_NEAR(_simulationBox->getMolecule(0).getAtomVelocity(0)[0], velocity_mol1_atom1[0], 10.0);
    EXPECT_NEAR(_simulationBox->getMolecule(0).getAtomVelocity(0)[1], velocity_mol1_atom1[1], 10.0);
    EXPECT_NEAR(_simulationBox->getMolecule(0).getAtomVelocity(0)[2], velocity_mol1_atom1[2], 10.0);
    EXPECT_NEAR(_simulationBox->getMolecule(0).getAtomVelocity(1)[0], velocity_mol1_atom2[0], 10.0);
    EXPECT_NEAR(_simulationBox->getMolecule(0).getAtomVelocity(1)[1], velocity_mol1_atom2[1], 10.0);
    EXPECT_NEAR(_simulationBox->getMolecule(0).getAtomVelocity(1)[2], velocity_mol1_atom2[2], 10.0);
    EXPECT_NEAR(_simulationBox->getMolecule(1).getAtomVelocity(0)[0], velocity_mol2_atom1[0], 10.0);
    EXPECT_NEAR(_simulationBox->getMolecule(1).getAtomVelocity(0)[1], velocity_mol2_atom1[1], 10.0);
    EXPECT_NEAR(_simulationBox->getMolecule(1).getAtomVelocity(0)[2], velocity_mol2_atom1[2], 10.0);
}

TEST_F(TestResetKinetics, noReset) { EXPECT_NO_THROW(_resetKinetics->reset(10, *_data, *_simulationBox)); }

TEST_F(TestResetKinetics, resetTemperautreNscale)
{
    _resetKinetics = new resetKinetics::ResetTemperature(10, 11, 0, 11, 300.0);

    const auto velocity_mol1_atom1_old = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_old = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_old = _simulationBox->getMolecule(1).getAtomVelocity(0);

    _data->setTemperature(100.0);

    _resetKinetics->reset(9, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NE(velocity_mol1_atom1_new[0], velocity_mol1_atom1_old[0]);
    EXPECT_NE(velocity_mol1_atom1_new[1], velocity_mol1_atom1_old[1]);
    EXPECT_NE(velocity_mol1_atom1_new[2], velocity_mol1_atom1_old[2]);
    EXPECT_NE(velocity_mol1_atom2_new[0], velocity_mol1_atom2_old[0]);
    EXPECT_NE(velocity_mol1_atom2_new[1], velocity_mol1_atom2_old[1]);
    EXPECT_NE(velocity_mol1_atom2_new[2], velocity_mol1_atom2_old[2]);
    EXPECT_NE(velocity_mol2_atom1_new[0], velocity_mol2_atom1_old[0]);
    EXPECT_NE(velocity_mol2_atom1_new[1], velocity_mol2_atom1_old[1]);
    EXPECT_NE(velocity_mol2_atom1_new[2], velocity_mol2_atom1_old[2]);

    _resetKinetics->reset(15, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new2 = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new2 = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new2 = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NEAR(velocity_mol1_atom1_new2[0], velocity_mol1_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[1], velocity_mol1_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[2], velocity_mol1_atom1_new[2], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[0], velocity_mol1_atom2_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[1], velocity_mol1_atom2_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[2], velocity_mol1_atom2_new[2], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[0], velocity_mol2_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[1], velocity_mol2_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[2], velocity_mol2_atom1_new[2], 10.0);
}

TEST_F(TestResetKinetics, resetTemperautreFscale)
{
    _resetKinetics = new resetKinetics::ResetTemperature(0, 9, 0, 11, 300.0);

    const auto velocity_mol1_atom1_old = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_old = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_old = _simulationBox->getMolecule(1).getAtomVelocity(0);

    _data->setTemperature(100.0);

    _resetKinetics->reset(9, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NE(velocity_mol1_atom1_new[0], velocity_mol1_atom1_old[0]);
    EXPECT_NE(velocity_mol1_atom1_new[1], velocity_mol1_atom1_old[1]);
    EXPECT_NE(velocity_mol1_atom1_new[2], velocity_mol1_atom1_old[2]);
    EXPECT_NE(velocity_mol1_atom2_new[0], velocity_mol1_atom2_old[0]);
    EXPECT_NE(velocity_mol1_atom2_new[1], velocity_mol1_atom2_old[1]);
    EXPECT_NE(velocity_mol1_atom2_new[2], velocity_mol1_atom2_old[2]);
    EXPECT_NE(velocity_mol2_atom1_new[0], velocity_mol2_atom1_old[0]);
    EXPECT_NE(velocity_mol2_atom1_new[1], velocity_mol2_atom1_old[1]);
    EXPECT_NE(velocity_mol2_atom1_new[2], velocity_mol2_atom1_old[2]);

    _resetKinetics->reset(15, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new2 = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new2 = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new2 = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NEAR(velocity_mol1_atom1_new2[0], velocity_mol1_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[1], velocity_mol1_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[2], velocity_mol1_atom1_new[2], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[0], velocity_mol1_atom2_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[1], velocity_mol1_atom2_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[2], velocity_mol1_atom2_new[2], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[0], velocity_mol2_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[1], velocity_mol2_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[2], velocity_mol2_atom1_new[2], 10.0);
}

TEST_F(TestResetKinetics, resetMomentumNreset)
{
    _resetKinetics = new resetKinetics::ResetMomentum(0, 11, 10, 11, 300.0);

    const auto velocity_mol1_atom1_old = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_old = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_old = _simulationBox->getMolecule(1).getAtomVelocity(0);

    _data->setMomentumVector(linearAlgebra::Vec3D(1.0, 2.0, 3.0));

    _resetKinetics->reset(9, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NE(velocity_mol1_atom1_new[0], velocity_mol1_atom1_old[0]);
    EXPECT_NE(velocity_mol1_atom1_new[1], velocity_mol1_atom1_old[1]);
    EXPECT_NE(velocity_mol1_atom1_new[2], velocity_mol1_atom1_old[2]);
    EXPECT_NE(velocity_mol1_atom2_new[0], velocity_mol1_atom2_old[0]);
    EXPECT_NE(velocity_mol1_atom2_new[1], velocity_mol1_atom2_old[1]);
    EXPECT_NE(velocity_mol1_atom2_new[2], velocity_mol1_atom2_old[2]);
    EXPECT_NE(velocity_mol2_atom1_new[0], velocity_mol2_atom1_old[0]);
    EXPECT_NE(velocity_mol2_atom1_new[1], velocity_mol2_atom1_old[1]);
    EXPECT_NE(velocity_mol2_atom1_new[2], velocity_mol2_atom1_old[2]);

    _resetKinetics->reset(15, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new2 = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new2 = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new2 = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NEAR(velocity_mol1_atom1_new2[0], velocity_mol1_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[1], velocity_mol1_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[2], velocity_mol1_atom1_new[2], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[0], velocity_mol1_atom2_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[1], velocity_mol1_atom2_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[2], velocity_mol1_atom2_new[2], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[0], velocity_mol2_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[1], velocity_mol2_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[2], velocity_mol2_atom1_new[2], 10.0);
}

TEST_F(TestResetKinetics, resetTemperatureNreset)
{
    _resetKinetics = new resetKinetics::ResetTemperature(0, 11, 10, 11, 300.0);

    const auto velocity_mol1_atom1_old = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_old = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_old = _simulationBox->getMolecule(1).getAtomVelocity(0);

    _data->setMomentumVector(linearAlgebra::Vec3D(1.0, 2.0, 3.0));

    _resetKinetics->reset(9, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NE(velocity_mol1_atom1_new[0], velocity_mol1_atom1_old[0]);
    EXPECT_NE(velocity_mol1_atom1_new[1], velocity_mol1_atom1_old[1]);
    EXPECT_NE(velocity_mol1_atom1_new[2], velocity_mol1_atom1_old[2]);
    EXPECT_NE(velocity_mol1_atom2_new[0], velocity_mol1_atom2_old[0]);
    EXPECT_NE(velocity_mol1_atom2_new[1], velocity_mol1_atom2_old[1]);
    EXPECT_NE(velocity_mol1_atom2_new[2], velocity_mol1_atom2_old[2]);
    EXPECT_NE(velocity_mol2_atom1_new[0], velocity_mol2_atom1_old[0]);
    EXPECT_NE(velocity_mol2_atom1_new[1], velocity_mol2_atom1_old[1]);
    EXPECT_NE(velocity_mol2_atom1_new[2], velocity_mol2_atom1_old[2]);

    _resetKinetics->reset(15, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new2 = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new2 = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new2 = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NEAR(velocity_mol1_atom1_new2[0], velocity_mol1_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[1], velocity_mol1_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[2], velocity_mol1_atom1_new[2], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[0], velocity_mol1_atom2_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[1], velocity_mol1_atom2_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[2], velocity_mol1_atom2_new[2], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[0], velocity_mol2_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[1], velocity_mol2_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[2], velocity_mol2_atom1_new[2], 10.0);
}

TEST_F(TestResetKinetics, resetMomentumFreset)
{
    _resetKinetics = new resetKinetics::ResetMomentum(0, 11, 0, 9, 300.0);

    const auto velocity_mol1_atom1_old = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_old = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_old = _simulationBox->getMolecule(1).getAtomVelocity(0);

    _data->setMomentumVector(linearAlgebra::Vec3D(1.0, 2.0, 3.0));

    _resetKinetics->reset(9, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NE(velocity_mol1_atom1_new[0], velocity_mol1_atom1_old[0]);
    EXPECT_NE(velocity_mol1_atom1_new[1], velocity_mol1_atom1_old[1]);
    EXPECT_NE(velocity_mol1_atom1_new[2], velocity_mol1_atom1_old[2]);
    EXPECT_NE(velocity_mol1_atom2_new[0], velocity_mol1_atom2_old[0]);
    EXPECT_NE(velocity_mol1_atom2_new[1], velocity_mol1_atom2_old[1]);
    EXPECT_NE(velocity_mol1_atom2_new[2], velocity_mol1_atom2_old[2]);
    EXPECT_NE(velocity_mol2_atom1_new[0], velocity_mol2_atom1_old[0]);
    EXPECT_NE(velocity_mol2_atom1_new[1], velocity_mol2_atom1_old[1]);
    EXPECT_NE(velocity_mol2_atom1_new[2], velocity_mol2_atom1_old[2]);

    _resetKinetics->reset(15, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new2 = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new2 = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new2 = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NEAR(velocity_mol1_atom1_new2[0], velocity_mol1_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[1], velocity_mol1_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[2], velocity_mol1_atom1_new[2], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[0], velocity_mol1_atom2_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[1], velocity_mol1_atom2_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[2], velocity_mol1_atom2_new[2], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[0], velocity_mol2_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[1], velocity_mol2_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[2], velocity_mol2_atom1_new[2], 10.0);
}

TEST_F(TestResetKinetics, resetTemperatureFreset)
{
    _resetKinetics = new resetKinetics::ResetTemperature(0, 11, 0, 9, 300.0);

    const auto velocity_mol1_atom1_old = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_old = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_old = _simulationBox->getMolecule(1).getAtomVelocity(0);

    _data->setMomentumVector(linearAlgebra::Vec3D(1.0, 2.0, 3.0));

    _resetKinetics->reset(9, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NE(velocity_mol1_atom1_new[0], velocity_mol1_atom1_old[0]);
    EXPECT_NE(velocity_mol1_atom1_new[1], velocity_mol1_atom1_old[1]);
    EXPECT_NE(velocity_mol1_atom1_new[2], velocity_mol1_atom1_old[2]);
    EXPECT_NE(velocity_mol1_atom2_new[0], velocity_mol1_atom2_old[0]);
    EXPECT_NE(velocity_mol1_atom2_new[1], velocity_mol1_atom2_old[1]);
    EXPECT_NE(velocity_mol1_atom2_new[2], velocity_mol1_atom2_old[2]);
    EXPECT_NE(velocity_mol2_atom1_new[0], velocity_mol2_atom1_old[0]);
    EXPECT_NE(velocity_mol2_atom1_new[1], velocity_mol2_atom1_old[1]);
    EXPECT_NE(velocity_mol2_atom1_new[2], velocity_mol2_atom1_old[2]);

    _resetKinetics->reset(15, *_data, *_simulationBox);

    const auto velocity_mol1_atom1_new2 = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2_new2 = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto velocity_mol2_atom1_new2 = _simulationBox->getMolecule(1).getAtomVelocity(0);

    EXPECT_NEAR(velocity_mol1_atom1_new2[0], velocity_mol1_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[1], velocity_mol1_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom1_new2[2], velocity_mol1_atom1_new[2], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[0], velocity_mol1_atom2_new[0], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[1], velocity_mol1_atom2_new[1], 10.0);
    EXPECT_NEAR(velocity_mol1_atom2_new2[2], velocity_mol1_atom2_new[2], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[0], velocity_mol2_atom1_new[0], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[1], velocity_mol2_atom1_new[1], 10.0);
    EXPECT_NEAR(velocity_mol2_atom1_new2[2], velocity_mol2_atom1_new[2], 10.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}