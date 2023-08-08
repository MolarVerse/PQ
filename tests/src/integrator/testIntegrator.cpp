#include "testIntegrator.hpp"

#include "constants.hpp"

/**
 * @brief tests function integrate velocities of velocity verlet integrator
 *
 */
TEST_F(TestIntegrator, integrateVelocities)
{
    _integrator->integrateVelocities(*_molecule1, 0);
    EXPECT_EQ(_molecule1->getAtomVelocity(0), linearAlgebra::Vec3D(0.0, 0.0, 0.0));
    _integrator->integrateVelocities(*_molecule1, 1);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomVelocity(1)[0], 1.0 + 0.1 * 0.5 * constants::_V_VERLET_VELOCITY_FACTOR_);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomVelocity(1)[1], 2.0 + 0.1 * 1.5 * constants::_V_VERLET_VELOCITY_FACTOR_);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomVelocity(1)[2], 3.0 + 0.1 * 2.5 * constants::_V_VERLET_VELOCITY_FACTOR_);
}

/**
 * @brief tests function integrate positions of velocity verlet integrator
 *
 */
TEST_F(TestIntegrator, integratePositions)
{
    _integrator->integratePositions(*_molecule1, 0, *_box);
    EXPECT_EQ(_molecule1->getAtomPosition(0), linearAlgebra::Vec3D(0.0, 0.0, 0.0));
    _integrator->integratePositions(*_molecule1, 1, *_box);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomPosition(1)[0], 1.0 + 0.1 * 1.0 * constants::_FS_TO_S_);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomPosition(1)[1], 1.0 + 0.1 * 2.0 * constants::_FS_TO_S_);
    EXPECT_DOUBLE_EQ(_molecule1->getAtomPosition(1)[2], 1.0 + 0.1 * 3.0 * constants::_FS_TO_S_);
}

/**
 * @brief tests function firstStep of velocity verlet integrator
 *
 */
TEST_F(TestIntegrator, firstStep)
{
    _integrator->firstStep(*_box);

    const auto molecule = _box->getMolecules()[0];
    EXPECT_EQ(molecule.getAtomVelocity(0), linearAlgebra::Vec3D(0.0, 0.0, 0.0));

    auto velocities  = linearAlgebra::Vec3D(1.0, 2.0, 3.0);
    velocities      += 0.1 * linearAlgebra::Vec3D(0.5, 1.5, 2.5) * constants::_V_VERLET_VELOCITY_FACTOR_;

    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[0], velocities[0]);
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[1], velocities[1]);
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[2], velocities[2]);

    EXPECT_DOUBLE_EQ(molecule.getAtomPosition(0)[0], 0.0);

    EXPECT_DOUBLE_EQ(molecule.getAtomPosition(1)[0], 1.0 + 0.1 * velocities[0] * constants::_FS_TO_S_);
    EXPECT_DOUBLE_EQ(molecule.getAtomPosition(1)[1], 1.0 + 0.1 * velocities[1] * constants::_FS_TO_S_);
    EXPECT_DOUBLE_EQ(molecule.getAtomPosition(1)[2], 1.0 + 0.1 * velocities[2] * constants::_FS_TO_S_);

    EXPECT_EQ(molecule.getAtomForce(0), linearAlgebra::Vec3D(0.0, 0.0, 0.0));
    EXPECT_EQ(molecule.getAtomForce(1), linearAlgebra::Vec3D(0.0, 0.0, 0.0));

    EXPECT_TRUE(molecule.getCenterOfMass() != linearAlgebra::Vec3D(0.0, 0.0, 0.0));
}

/**
 * @brief tests function secondStep of velocity verlet integrator
 *
 */
TEST_F(TestIntegrator, secondStep)
{
    _integrator->secondStep(*_box);

    const auto molecule = _box->getMolecules()[0];
    EXPECT_EQ(molecule.getAtomVelocity(0), linearAlgebra::Vec3D(0.0, 0.0, 0.0));
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[0], 1.0 + 0.1 * 0.5 * constants::_V_VERLET_VELOCITY_FACTOR_);
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[1], 2.0 + 0.1 * 1.5 * constants::_V_VERLET_VELOCITY_FACTOR_);
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[2], 3.0 + 0.1 * 2.5 * constants::_V_VERLET_VELOCITY_FACTOR_);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}