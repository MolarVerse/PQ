#include "testPostProcessSetup.hpp"

using namespace std;
using namespace ::testing;

TEST_F(TestPostProcessSetup, testSetAtomMasses)
{
    Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    molecule.addAtomName("C");
    molecule.addAtomName("H");
    molecule.addAtomName("O");

    _engine.getSimulationBox()._molecules.push_back(molecule);
    PostProcessSetup postProcessSetup(_engine);
    postProcessSetup.setAtomMasses();

    EXPECT_EQ(_engine.getSimulationBox()._molecules[0].getMass(0), 12.0107);
    cout << "__DEBUG__" << _engine.getSimulationBox()._molecules[0].getNumberOfAtoms() << endl;
    EXPECT_EQ(_engine.getSimulationBox()._molecules[0].getMass(1), 1.0078);
    cout << "__DEBUG__" << endl;
    EXPECT_EQ(_engine.getSimulationBox()._molecules[0].getMass(2), 15.9994);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}