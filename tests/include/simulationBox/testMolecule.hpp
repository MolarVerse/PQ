#ifndef _TEST_MOLECULE_HPP_

#define _TEST_MOLECULE_HPP_

#include "atom.hpp"       // for Atom
#include "molecule.hpp"   // for Molecule
#include "vector3d.hpp"   // for Vec3D

#include <gtest/gtest.h>   // for Test
#include <memory>          // for __shared_ptr_access, shared_ptr, make_shared

class TestMolecule : public ::testing::Test
{
  protected:
    virtual void SetUp()
    {
        _molecule = new simulationBox::Molecule();
        _molecule->setNumberOfAtoms(3);

        auto _atom1 = std::make_shared<simulationBox::Atom>();
        auto _atom2 = std::make_shared<simulationBox::Atom>();
        auto _atom3 = std::make_shared<simulationBox::Atom>();

        _atom1->setExternalAtomType(1);
        _atom2->setExternalAtomType(2);
        _atom3->setExternalAtomType(2);

        _atom1->setPosition(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        _atom2->setPosition(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        _atom3->setPosition(linearAlgebra::Vec3D(0.0, 1.0, 0.0));

        _atom1->setVelocity(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        _atom2->setVelocity(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        _atom3->setVelocity(linearAlgebra::Vec3D(0.0, 1.0, 0.0));

        _atom1->setForce(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        _atom2->setForce(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        _atom3->setForce(linearAlgebra::Vec3D(0.0, 1.0, 0.0));

        _atom1->setMass(1.0);
        _atom2->setMass(2.0);
        _atom3->setMass(3.0);

        _molecule->setMolMass(6.0);

        _molecule->addAtom(_atom1);
        _molecule->addAtom(_atom2);
        _molecule->addAtom(_atom3);
    }

    virtual void TearDown() { delete _molecule; }

    simulationBox::Molecule *_molecule;
};

#endif   // _TEST_MOLECULE_HPP_