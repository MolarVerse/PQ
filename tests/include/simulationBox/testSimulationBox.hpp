#ifndef _TEST_SIMULATION_BOX_HPP_

#define _TEST_SIMULATION_BOX_HPP_

#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for Vec3D

#include <gtest/gtest.h>   // for Test

class TestSimulationBox : public ::testing::Test
{
  protected:
    virtual void SetUp()
    {
        _simulationBox = new simulationBox::SimulationBox();

        auto molecule1 = simulationBox::Molecule();
        auto molecule2 = simulationBox::Molecule();

        molecule1.setNumberOfAtoms(3);
        molecule2.setNumberOfAtoms(2);

        auto atom1 = std::make_shared<simulationBox::Atom>();
        auto atom2 = std::make_shared<simulationBox::Atom>();
        auto atom3 = std::make_shared<simulationBox::Atom>();

        atom1->setPosition(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        atom2->setPosition(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        atom3->setPosition(linearAlgebra::Vec3D(0.0, 1.0, 0.0));
        atom1->setMass(1.0);
        atom2->setMass(2.0);
        atom3->setMass(3.0);

        molecule1.setMolMass(6.0);
        molecule1.setMoltype(1);
        molecule1.addAtom(atom1);
        molecule1.addAtom(atom2);
        molecule1.addAtom(atom3);

        auto atom4 = std::make_shared<simulationBox::Atom>();
        auto atom5 = std::make_shared<simulationBox::Atom>();

        atom4->setPosition(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        atom5->setPosition(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        atom4->setMass(1.0);
        atom5->setMass(2.0);

        molecule2.setMolMass(3.0);
        molecule2.setMoltype(2);
        molecule2.addAtom(atom4);
        molecule2.addAtom(atom5);

        _simulationBox->addMolecule(molecule1);
        _simulationBox->addMolecule(molecule2);

        auto moleculeType1 = simulationBox::MoleculeType(1);
        auto moleculeType2 = simulationBox::MoleculeType(2);

        _simulationBox->addMoleculeType(moleculeType1);
        _simulationBox->addMoleculeType(moleculeType2);
        _simulationBox->addAtom(atom1);
        _simulationBox->addAtom(atom2);
        _simulationBox->addAtom(atom3);
        _simulationBox->addAtom(atom4);
        _simulationBox->addAtom(atom5);

        _simulationBox->setBoxDimensions({10.0, 10.0, 10.0});
    }

    virtual void TearDown() { delete _simulationBox; }

    simulationBox::SimulationBox *_simulationBox;
};

#endif   // _TEST_SIMULATION_BOX_HPP_