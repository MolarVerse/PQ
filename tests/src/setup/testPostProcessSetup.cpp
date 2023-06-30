#include "testPostProcessSetup.hpp"

#include "constants.hpp"
#include "exceptions.hpp"

#include <cmath>

using namespace std;
using namespace ::testing;
using namespace setup;
using namespace simulationBox;
using namespace config;
using namespace customException;

TEST_F(TestPostProcessSetup, testSetAtomMasses)
{
    Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    molecule.addAtomName("C");
    molecule.addAtomName("H");
    molecule.addAtomName("O");

    _engine.getSimulationBox().getMolecules().push_back(molecule);
    PostProcessSetup postProcessSetup(_engine);
    postProcessSetup.setAtomMasses();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomMass(0), 12.0107);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomMass(1), 1.00794);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomMass(2), 15.9994);
}

TEST_F(TestPostProcessSetup, testSetAtomMassesThrowsError)
{
    Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    molecule.addAtomName("H");
    molecule.addAtomName("D");
    molecule.addAtomName("NOTANATOMNAME");

    _engine.getSimulationBox().getMolecules().push_back(molecule);
    PostProcessSetup postProcessSetup(_engine);
    ASSERT_THROW(postProcessSetup.setAtomMasses(), MolDescriptorException);
}

TEST_F(TestPostProcessSetup, testSetAtomicNumbers)
{
    Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    molecule.addAtomName("C");
    molecule.addAtomName("H");
    molecule.addAtomName("O");

    _engine.getSimulationBox().getMolecules().push_back(molecule);
    PostProcessSetup postProcessSetup(_engine);
    postProcessSetup.setAtomicNumbers();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomicNumber(0), 6);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomicNumber(1), 1);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomicNumber(2), 8);
}

TEST_F(TestPostProcessSetup, testSetAtomicNumbersThrowsError)
{
    Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    molecule.addAtomName("H");
    molecule.addAtomName("D");
    molecule.addAtomName("NOTANATOMNAME");

    _engine.getSimulationBox().getMolecules().push_back(molecule);
    PostProcessSetup postProcessSetup(_engine);
    ASSERT_THROW(postProcessSetup.setAtomicNumbers(), MolDescriptorException);
}

TEST_F(TestPostProcessSetup, testSetTotalMass)
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
    PostProcessSetup postProcessSetup(_engine);
    postProcessSetup.setAtomMasses();
    postProcessSetup.calculateTotalMass();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getTotalMass(), 12.0107 + 3 * 1.00794 + 15.9994);
}

TEST_F(TestPostProcessSetup, testSetMolMass)
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
    PostProcessSetup postProcessSetup(_engine);
    postProcessSetup.setAtomMasses();
    postProcessSetup.calculateMolMass();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getMolecules()[0].getMolMass(), 12.0107 + 1 * 1.00794 + 15.9994);
}

TEST_F(TestPostProcessSetup, testSetTotalCharge)
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
    PostProcessSetup postProcessSetup(_engine);
    postProcessSetup.calculateTotalCharge();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getTotalCharge(), -0.1);
}

TEST_F(TestPostProcessSetup, testNoDensityNoBox)
{
    PostProcessSetup postProcessSetup(_engine);
    ASSERT_THROW(postProcessSetup.checkBoxSettings(), UserInputException);
}

TEST_F(TestPostProcessSetup, testNoDensity)
{
    _engine.getSimulationBox().setTotalMass(6000);
    _engine.getSimulationBox().setBoxDimensions({10.0, 20.0, 30.0});
    PostProcessSetup postProcessSetup(_engine);
    postProcessSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getVolume(), 6000.0);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getDensity(), _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_);
}

TEST_F(TestPostProcessSetup, testNoBox)
{
    _engine.getSimulationBox().setTotalMass(6000);
    _engine.getSimulationBox().setDensity(_AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_);
    PostProcessSetup postProcessSetup(_engine);
    postProcessSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getVolume(), 6000.0);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getBoxDimensions()[2], cbrt(6000.0));
}

TEST_F(TestPostProcessSetup, testBoxAndDensitySet)
{
    _engine.getSimulationBox().setTotalMass(6000);
    _engine.getSimulationBox().setDensity(12341243.1234);   // this should be ignored
    _engine.getSimulationBox().setBoxDimensions({10.0, 20.0, 30.0});
    PostProcessSetup postProcessSetup(_engine);
    postProcessSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getVolume(), 6000.0);
    EXPECT_DOUBLE_EQ(_engine.getSimulationBox().getDensity(), _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_);
}

TEST_F(TestPostProcessSetup, testResizeAtomShiftForces)
{

    Molecule molecule1(1);
    molecule1.addAtomForce({1.0, 2.0, 3.0});
    molecule1.addAtomForce({4.0, 5.0, 6.0});

    Molecule molecule2(2);
    molecule2.addAtomForce({7.0, 8.0, 9.0});

    _engine.getSimulationBox().addMolecule(molecule1);
    _engine.getSimulationBox().addMolecule(molecule2);

    PostProcessSetup postProcessSetup(_engine);
    postProcessSetup.resizeAtomShiftForces();

    EXPECT_EQ(_engine.getSimulationBox().getMolecules()[0].getAtomShiftForces().size(), 2);
    EXPECT_EQ(_engine.getSimulationBox().getMolecules()[1].getAtomShiftForces().size(), 1);
}

TEST_F(TestPostProcessSetup, testChechRcCutoff)
{
    _engine.getSimulationBox().setBoxDimensions({10.0, 20.0, 30.0});
    _engine.getSimulationBox().setRcCutOff(14.0);
    PostProcessSetup postProcessSetup(_engine);
    EXPECT_THROW(postProcessSetup.checkRcCutoff(), customException::InputFileException);

    PostProcessSetup postProcessSetup2(_engine);
    _engine.getSimulationBox().setRcCutOff(4.0);
    EXPECT_NO_THROW(postProcessSetup2.checkRcCutoff());
}

TEST_F(TestPostProcessSetup, testSetTimeStep)
{
    _engine.getTimings().setTimestep(0.001);
    PostProcessSetup postProcessSetup(_engine);
    postProcessSetup.setTimestep();
    EXPECT_DOUBLE_EQ(_engine.getIntegrator().getDt(), 0.001);
}

// TEST_F(TestPostProcessSetup, testSetup)
// {
//     Molecule molecule1(1);
//     molecule1.setNumberOfAtoms(3);
//     molecule1.addAtomName("C");
//     molecule1.addAtomName("H");
//     molecule1.addAtomName("O");

//     Molecule molecule2(2);
//     molecule2.setNumberOfAtoms(2);
//     molecule2.addAtomName("H");
//     molecule2.addAtomName("H");

//     molecule1.addPartialCharge(0.1);
//     molecule1.addPartialCharge(0.2);
//     molecule1.addPartialCharge(-0.4);

//     molecule2.addPartialCharge(0.1);
//     molecule2.addPartialCharge(0.2);

//     _engine.getSimulationBox().getMolecules().push_back(molecule1);
//     _engine.getSimulationBox().getMolecules().push_back(molecule2);
//     _engine.getSimulationBox().setTotalMass(6000);
//     _engine.getSimulationBox().setDensity(12341243.1234);   // this should be ignored
//     _engine.getSimulationBox().setBoxDimensions({30.0, 35.0, 32.0});
//     ASSERT_NO_THROW(postProcessSetup(_engine));
// }

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}