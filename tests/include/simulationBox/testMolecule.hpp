#ifndef _TEST_MOLECULE_HPP_

#define _TEST_MOLECULE_HPP_

#include "molecule.hpp"

#include <gtest/gtest.h>

class TestMolecule : public ::testing::Test
{
  protected:
    virtual void SetUp()
    {
        _molecule = new simulationBox::Molecule();
        _molecule->setExternalAtomTypes({1, 2, 2});

        _molecule->addAtomPosition(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        _molecule->addAtomPosition(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        _molecule->addAtomPosition(linearAlgebra::Vec3D(0.0, 1.0, 0.0));

        _molecule->addAtomVelocity(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        _molecule->addAtomVelocity(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        _molecule->addAtomVelocity(linearAlgebra::Vec3D(0.0, 1.0, 0.0));

        _molecule->addAtomForce(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        _molecule->addAtomForce(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        _molecule->addAtomForce(linearAlgebra::Vec3D(0.0, 1.0, 0.0));

        _molecule->addAtomMass(1.0);
        _molecule->addAtomMass(2.0);
        _molecule->addAtomMass(3.0);

        _molecule->setMolMass(6.0);

        _molecule->setNumberOfAtoms(3);
    }

    virtual void TearDown() { delete _molecule; }

    simulationBox::Molecule *_molecule;
};

#endif   // _TEST_MOLECULE_HPP_