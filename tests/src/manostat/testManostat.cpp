#include "testManostat.hpp"

#include "constants.hpp"

using namespace std;

TEST_F(TestManostat, CalculatePressure)
{
    _manostat->calculatePressure(*_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * constants::_PRESSURE_FACTOR_);
}

TEST_F(TestManostat, ChangeVirialToAtomic)
{
    _data->changeKineticVirialToAtomic();

    _manostat->calculatePressure(*_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 2.0 * constants::_PRESSURE_FACTOR_);
}

TEST_F(TestManostat, testApplyBerendsenManostat)
{
    _box->setBoxDimensions({2.0, 2.0, 2.0});
    auto box_old = _box->getBoxDimensions();

    auto molecule = simulationBox::Molecule();
    molecule.addAtomPosition({1.0, 0.0, 0.0});
    molecule.setCenterOfMass({1.0, 0.0, 0.0});
    molecule.setNumberOfAtoms(1);

    _box->addMolecule(molecule);

    _manostat = new manostat::BerendsenManostat(1.0, 0.1, 4.5);
    _manostat->applyManostat(*_box, *_data);

    auto box_new = _box->getBoxDimensions();

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * constants::_PRESSURE_FACTOR_);
    EXPECT_NE(box_old, box_new);
    EXPECT_NE(_box->getMolecule(0).getAtomPosition(0), linearAlgebra::Vec3D(1.0, 0.0, 0.0));
}

TEST_F(TestManostat, applyNoneManostat)
{
    _manostat->applyManostat(*_box, *_data);

    EXPECT_DOUBLE_EQ(_data->getPressure(), 3.0 * constants::_PRESSURE_FACTOR_);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}