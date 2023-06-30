#include "testIntegrator.hpp"

#include "constants.hpp"

TEST_F(TestIntegrator, integrateVelocities)
{
    _integrator->integrateVelocities(*_molecule1, 0);
    EXPECT_EQ(_molecule1->getAtomVelocity(0), vector3d::Vec3D(0.0, 0.0, 0.0));
    _integrator->integrateVelocities(*_molecule1, 1);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomVelocity(1)[0], 1.0 + 0.1 * 0.5 * config::_V_VERLET_VELOCITY_FACTOR_);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomVelocity(1)[1], 2.0 + 0.1 * 1.5 * config::_V_VERLET_VELOCITY_FACTOR_);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomVelocity(1)[2], 3.0 + 0.1 * 2.5 * config::_V_VERLET_VELOCITY_FACTOR_);
}

TEST_F(TestIntegrator, integratePositions)
{
    _integrator->integratePositions(*_molecule1, 0, *_box);
    EXPECT_EQ(_molecule1->getAtomPosition(0), vector3d::Vec3D(0.0, 0.0, 0.0));
    _integrator->integratePositions(*_molecule1, 1, *_box);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomPosition(1)[0], 1.0 + 0.1 * 1.0 * config::_FS_TO_S_);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomPosition(1)[1], 1.0 + 0.1 * 2.0 * config::_FS_TO_S_);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomPosition(1)[2], 1.0 + 0.1 * 3.0 * config::_FS_TO_S_);
}

TEST_F(TestIntegrator, firstStep)
{
    _integrator->firstStep(*_box);

    const auto molecule = _box->getMolecules()[0];
    EXPECT_EQ(molecule.getAtomVelocity(0), vector3d::Vec3D(0.0, 0.0, 0.0));

    auto velocities  = vector3d::Vec3D(1.0, 2.0, 3.0);
    velocities      += 0.1 * vector3d::Vec3D(0.5, 1.5, 2.5) * config::_V_VERLET_VELOCITY_FACTOR_;

    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[0], velocities[0]);
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[1], velocities[1]);
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[2], velocities[2]);

    EXPECT_DOUBLE_EQ(molecule.getAtomPosition(0)[0], 0.0);

    EXPECT_DOUBLE_EQ(molecule.getAtomPosition(1)[0], 1.0 + 0.1 * velocities[0] * config::_FS_TO_S_);
    EXPECT_DOUBLE_EQ(molecule.getAtomPosition(1)[1], 1.0 + 0.1 * velocities[1] * config::_FS_TO_S_);
    EXPECT_DOUBLE_EQ(molecule.getAtomPosition(1)[2], 1.0 + 0.1 * velocities[2] * config::_FS_TO_S_);

    EXPECT_EQ(molecule.getAtomForce(0), vector3d::Vec3D(0.0, 0.0, 0.0));
    EXPECT_EQ(molecule.getAtomForce(1), vector3d::Vec3D(0.0, 0.0, 0.0));

    EXPECT_TRUE(molecule.getCenterOfMass() != vector3d::Vec3D(0.0, 0.0, 0.0));
}

TEST_F(TestIntegrator, secondStep)
{
    _integrator->secondStep(*_box);

    const auto molecule = _box->getMolecules()[0];
    EXPECT_EQ(molecule.getAtomVelocity(0), vector3d::Vec3D(0.0, 0.0, 0.0));
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[0], 1.0 + 0.1 * 0.5 * config::_V_VERLET_VELOCITY_FACTOR_);
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[1], 2.0 + 0.1 * 1.5 * config::_V_VERLET_VELOCITY_FACTOR_);
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[2], 3.0 + 0.1 * 2.5 * config::_V_VERLET_VELOCITY_FACTOR_);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}