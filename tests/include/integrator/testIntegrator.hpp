#ifndef _TEST_INTEGRATOR_HPP_

#define _TEST_INTEGRATOR_HPP_

#include "integrator.hpp"        // for Integrator, VelocityVerlet
#include "molecule.hpp"          // for Molecule
#include "simulationBox.hpp"     // for SimulationBox
#include "timingsSettings.hpp"   // for TimingsSettings
#include "vector3d.hpp"          // for Vec3D

#include <gtest/gtest.h>   // for Test

/**
 * class TestIntegrator
 *
 * @brief Fixture for integrator tests.
 *
 */
class TestIntegrator : public ::testing::Test
{
  protected:
    virtual void SetUp()
    {
        _integrator = new integrator::VelocityVerlet();
        settings::TimingsSettings::setTimeStep(0.1);

        _molecule1 = new simulationBox::Molecule();
        _molecule1->setNumberOfAtoms(2);

        auto atom1 = std::make_shared<simulationBox::Atom>();
        auto atom2 = std::make_shared<simulationBox::Atom>();

        atom1->setPosition(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        atom2->setPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));

        atom1->setVelocity(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        atom2->setVelocity(linearAlgebra::Vec3D(1.0, 2.0, 3.0));

        atom1->setForce(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        atom2->setForce(linearAlgebra::Vec3D(1.0, 3.0, 5.0));

        atom1->setMass(1.0);
        atom2->setMass(2.0);

        _molecule1->addAtom(atom1);
        _molecule1->addAtom(atom2);

        _molecule1->setMolMass(3.0);

        _box = new simulationBox::SimulationBox();
        _box->setBoxDimensions(linearAlgebra::Vec3D(10.0, 10.0, 10.0));

        _box->addMolecule(*_molecule1);
    }

    virtual void TearDown()
    {
        delete _integrator;
        delete _molecule1;
        delete _box;
    }

    integrator::Integrator       *_integrator;
    simulationBox::Molecule      *_molecule1;
    simulationBox::SimulationBox *_box;
};

#endif   // _TEST_INTEGRATOR_HPP_