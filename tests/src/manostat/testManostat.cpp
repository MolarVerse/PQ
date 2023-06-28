#include "testManostat.hpp"

#include "constants.hpp"

using namespace std;

TEST_F(TestManostat, testCalculatePressure)
{
    _manostat.calculatePressure(*_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * config::_PRESSURE_FACTOR_);
}

TEST_F(TestManostat, testChangeVirialToAtomic)
{
    _data->changeKineticVirialToAtomic();

    _manostat.calculatePressure(*_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 2.0 * config::_PRESSURE_FACTOR_);
}

TEST_F(TestManostat, testApplyManostat)
{
    _manostat.applyManostat(*_box, *_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * config::_PRESSURE_FACTOR_);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}