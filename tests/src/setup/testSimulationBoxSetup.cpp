#include "constants.hpp"
#include "exceptions.hpp"
#include "simulationBoxSetup.hpp"
#include "testSetup.hpp"

#include <cmath>

using namespace std;
using namespace ::testing;
using namespace setup;
using namespace simulationBox;
using namespace customException;

TEST_F(TestSetup, testSetAtomMasses)
{
    Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    molecule.addAtomName("C");
    molecule.addAtomName("H");
    molecule.addAtomName("O");

    _engine.getSimulationBox().getMolecules().push_back(molecule);
    SimulationBoxSetup simulationBoxSetup(_engine);
    simulationBoxSetup.setAtomMasses();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomMass(0), 12.0107);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomMass(1), 1.00794);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomMass(2), 15.9994);
}

TEST_F(TestSetup, testSetAtomMassesThrowsError)
{
    Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    molecule.addAtomName("H");
    molecule.addAtomName("D");
    molecule.addAtomName("NOTANATOMNAME");

    _engine.getSimulationBox().getMolecules().push_back(molecule);
    SimulationBoxSetup simulationBoxSetup(_engine);
    ASSERT_THROW(simulationBoxSetup.setAtomMasses(), MolDescriptorException);
}

TEST_F(TestSetup, testSetAtomicNumbers)
{
    Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    molecule.addAtomName("C");
    molecule.addAtomName("H");
    molecule.addAtomName("O");

    _engine.getSimulationBox().getMolecules().push_back(molecule);
    SimulationBoxSetup simulationBoxSetup(_engine);
    simulationBoxSetup.setAtomicNumbers();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomicNumber(0), 6);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomicNumber(1), 1);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomicNumber(2), 8);
}

TEST_F(TestSetup, testSetAtomicNumbersThrowsError)
{
    Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    molecule.addAtomName("H");
    molecule.addAtomName("D");
    molecule.addAtomName("NOTANATOMNAME");

    _engine.getSimulationBox().getMolecules().push_back(molecule);
    SimulationBoxSetup simulationBoxSetup(_engine);
    ASSERT_THROW(simulationBoxSetup.setAtomicNumbers(), MolDescriptorException);
}

TEST_F(TestSetup, testSetTotalMass)
{
    Molecule molecule1(1);
    molecule1.setNumberOfAtoms(3);
    molecule1.addAtomName("C");
    molecule1.addAtomName("H");
    molecule1.addAtomName("O");

    Molecule molecule2(2);
    molecule2.setNumberOfAtoms(2);
    molecule2.addAtomName("H");
    molecule2.addAtomName("H");

    _engine.getSimulationBox().getMolecules().push_back(molecule1);
    _engine.getSimulationBox().getMolecules().push_back(molecule2);
    SimulationBoxSetup simulationBoxSetup(_engine);
    simulationBoxSetup.setAtomMasses();
    simulationBoxSetup.calculateTotalMass();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getTotalMass(), 12.0107 + 3 * 1.00794 + 15.9994);
}

TEST_F(TestSetup, testSetMolMass)
{
    Molecule molecule1(1);
    molecule1.setNumberOfAtoms(3);
    molecule1.addAtomName("C");
    molecule1.addAtomName("H");
    molecule1.addAtomName("O");

    Molecule molecule2(2);
    molecule2.setNumberOfAtoms(2);
    molecule2.addAtomName("H");
    molecule2.addAtomName("H");

    _engine.getSimulationBox().getMolecules().push_back(molecule1);
    _engine.getSimulationBox().getMolecules().push_back(molecule2);
    SimulationBoxSetup simulationBoxSetup(_engine);
    simulationBoxSetup.setAtomMasses();
    simulationBoxSetup.calculateMolMass();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getMolMass(), 12.0107 + 1 * 1.00794 + 15.9994);
}

TEST_F(TestSetup, testSetTotalCharge)
{
    Molecule molecule1(1);
    molecule1.setNumberOfAtoms(3);
    molecule1.addAtomName("C");
    molecule1.addAtomName("H");
    molecule1.addAtomName("O");

    molecule1.addPartialCharge(0.1);
    molecule1.addPartialCharge(0.2);
    molecule1.addPartialCharge(-0.4);

    _engine.getSimulationBox().getMolecules().push_back(molecule1);
    SimulationBoxSetup simulationBoxSetup(_engine);
    simulationBoxSetup.calculateTotalCharge();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getTotalCharge(), -0.1);
}

TEST_F(TestSetup, testNoDensityNoBox)
{
    SimulationBoxSetup simulationBoxSetup(_engine);
    ASSERT_THROW(simulationBoxSetup.checkBoxSettings(), UserInputException);
}

TEST_F(TestSetup, testNoDensity)
{
    _engine.getSimulationBox().setTotalMass(6000);
    _engine.getSimulationBox().setBoxDimensions({10.0, 20.0, 30.0});
    SimulationBoxSetup simulationBoxSetup(_engine);
    simulationBoxSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getVolume(), 6000.0);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getDensity(), constants::_AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_);
}

TEST_F(TestSetup, testNoBox)
{
    _engine.getSimulationBox().setTotalMass(6000);
    _engine.getSimulationBox().setDensity(constants::_AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_);
    SimulationBoxSetup simulationBoxSetup(_engine);
    simulationBoxSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getVolume(), 6000.0);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getBoxDimensions()[2], cbrt(6000.0));
}

TEST_F(TestSetup, testBoxAndDensitySet)
{
    _engine.getSimulationBox().setTotalMass(6000);
    _engine.getSimulationBox().setDensity(12341243.1234);   // this should be ignored
    _engine.getSimulationBox().setBoxDimensions({10.0, 20.0, 30.0});
    SimulationBoxSetup simulationBoxSetup(_engine);
    simulationBoxSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getVolume(), 6000.0);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getDensity(), constants::_AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_);
}

TEST_F(TestSetup, testResizeAtomShiftForces)
{

    Molecule molecule1(1);
    molecule1.addAtomForce({1.0, 2.0, 3.0});
    molecule1.addAtomForce({4.0, 5.0, 6.0});

    Molecule molecule2(2);
    molecule2.addAtomForce({7.0, 8.0, 9.0});

    _engine.getSimulationBox().addMolecule(molecule1);
    _engine.getSimulationBox().addMolecule(molecule2);

    SimulationBoxSetup simulationBoxSetup(_engine);
    simulationBoxSetup.resizeAtomShiftForces();

    EXPECT_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomShiftForces().size(), 2);
    EXPECT_EQ(_engine.getSimulationBox().getMolecules()[1].getAtomShiftForces().size(), 1);
}

TEST_F(TestSetup, testChechRcCutoff)
{
    _engine.getSimulationBox().setBoxDimensions({10.0, 20.0, 30.0});
    _engine.getSimulationBox().setRcCutOff(14.0);
    SimulationBoxSetup simulationBoxSetup(_engine);
    EXPECT_THROW(simulationBoxSetup.checkRcCutoff(), customException::InputFileException);

    SimulationBoxSetup simulationBox2Setup(_engine);
    _engine.getSimulationBox().setRcCutOff(4.0);
    EXPECT_NO_THROW(simulationBox2Setup.checkRcCutoff());
}

/**
 * @brief testing full setup of simulation box
 *
 * @TODO: this test is not complete, it only tests the functions that are called in the setup
 *
 */
TEST_F(TestSetup, testFullSetup)
{
    Molecule molecule1(1);
    molecule1.setNumberOfAtoms(3);
    molecule1.addAtomName("C");
    molecule1.addAtomName("H");
    molecule1.addAtomName("O");
    molecule1.addPartialCharge(0.1);
    molecule1.addPartialCharge(0.2);
    molecule1.addPartialCharge(-0.4);

    Molecule molecule2(2);
    molecule2.setNumberOfAtoms(2);
    molecule2.addAtomName("H");
    molecule2.addAtomName("H");
    molecule2.addPartialCharge(0.1);
    molecule2.addPartialCharge(0.2);

    _engine.getSimulationBox().getMolecules().push_back(molecule1);
    _engine.getSimulationBox().getMolecules().push_back(molecule2);

    _engine.getSimulationBox().setTotalMass(33.0);
    _engine.getSimulationBox().setDensity(12341243.1234);   // this should be ignored
    _engine.getSimulationBox().setBoxDimensions({10.0, 20.0, 30.0});
    _engine.getSimulationBox().setRcCutOff(4.0);

    EXPECT_NO_THROW(setupSimulationBox(_engine));
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}