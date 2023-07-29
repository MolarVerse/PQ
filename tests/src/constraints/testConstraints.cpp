#include "testConstraints.hpp"

/**
 * @brief tests calculation of all bond constraints ref bond lengths
 *
 */
TEST_F(TestConstraints, calcRefBondLengths)
{
    _constraints->calculateConstraintBondRefs(*_box);
    EXPECT_EQ(_constraints->getBondConstraints()[0].getShakeDistanceRef(), vector3d::Vec3D(0.0, -1.0, -2.0));
    EXPECT_EQ(_constraints->getBondConstraints()[1].getShakeDistanceRef(), vector3d::Vec3D(1.0, -2.0, -3.0));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}