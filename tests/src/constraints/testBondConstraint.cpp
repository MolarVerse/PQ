#include "testBondConstraint.hpp"

/**
 * @brief tests calculation of bond constraint ref bond length
 *
 */
TEST_F(TestBondConstraint, calcRefBondLength)
{
    _bondConstraint->calculateConstraintBondRef(*_box);
    EXPECT_EQ(_bondConstraint->getShakeDistanceRef(), vector3d::Vec3D(0.0, -1.0, -2.0));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}