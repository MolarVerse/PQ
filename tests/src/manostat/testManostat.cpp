#include "testManostat.hpp"

#include "constants.hpp"

using namespace std;

TEST_F(TestManostat, CalculatePressure)
{
    _manostat.calculatePressure(*_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * config::_PRESSURE_FACTOR_);
}

TEST_F(TestManostat, ChangeVirialToAtomic)
{
    _data->changeKineticVirialToAtomic();

    _manostat.calculatePressure(*_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 2.0 * config::_PRESSURE_FACTOR_);
}

TEST_F(TestManostat, testApplyManostat)
{
    _manostat = manostat::BerendsenManostat(1.0, 0.1);
    _manostat.applyManostat(*_box, *_data);

    const auto old_pressure = 2.0 * config::_PRESSURE_FACTOR_;

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * config::_PRESSURE_FACTOR_);
}

TEST_F(TestManostat, applyManostatBerendsen) {}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}