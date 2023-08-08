#ifndef _TEST_BOND_CONSTRAINT_HPP_

#define _TEST_BOND_CONSTRAINT_HPP_

#include "bondConstraint.hpp"
#include "molecule.hpp"
#include "simulationBox.hpp"

#include <gtest/gtest.h>

/**
 * @class TestBondConstraint
 *
 * @brief Fixture for bond constraint tests.
 *
 */
class TestBondConstraint : public ::testing::Test
{
  protected:
    virtual void SetUp()
    {
        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(3);

        molecule1.addAtomPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule1.addAtomPosition(linearAlgebra::Vec3D(1.0, 2.0, 3.0));

        molecule1.addAtomMass(1.0);
        molecule1.addAtomMass(2.0);

        molecule1.addAtomVelocity(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        molecule1.addAtomVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));

        _box = new simulationBox::SimulationBox();
        _box->addMolecule(molecule1);
        _box->setBoxDimensions(linearAlgebra::Vec3D(10.0, 10.0, 10.0));

        _bondConstraint =
            new constraints::BondConstraint(&(_box->getMolecules()[0]), &(_box->getMolecules()[0]), 0, 1, _targetBondLength);
    }

    virtual void TearDown()
    {
        delete _box;
        delete _bondConstraint;
    }

    simulationBox::SimulationBox *_box;
    constraints::BondConstraint  *_bondConstraint;
    double                        _targetBondLength = 1.2;
};

#endif   // _TEST_BOND_CONSTRAINT_HPP_