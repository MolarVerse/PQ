#ifndef _BOND_CONSTRAINT_HPP_

#define _BOND_CONSTRAINT_HPP_

#include "molecule.hpp"
#include "simulationBox.hpp"

namespace constraints
{
    class BondConstraint;
}

/**
 * @class BondConstraint
 *
 * @brief constraint object for single bond length
 *
 */
class constraints::BondConstraint
{
  private:
    simulationBox::Molecule *_molecule1;
    simulationBox::Molecule *_molecule2;

    size_t _atomIndex1;
    size_t _atomIndex2;

    double          _targetBondLength;
    vector3d::Vec3D _shakeDistanceRef;

  public:
    BondConstraint(simulationBox::Molecule *molecule1,
                   simulationBox::Molecule *molecule2,
                   size_t                   atomIndex1,
                   size_t                   atomIndex2,
                   double                   bondLength)
        : _molecule1(molecule1), _molecule2(molecule2), _atomIndex1(atomIndex1), _atomIndex2(atomIndex2),
          _targetBondLength(bondLength){};

    void setShakeDistanceRef(vector3d::Vec3D shakeDistanceRef) { _shakeDistanceRef = shakeDistanceRef; }

    void calculateConstraintBondRef(const simulationBox::SimulationBox &);

    double calculateDistanceDelta(const simulationBox::SimulationBox &) const;
    double calculateVelocityDelta() const;
    bool   applyShake(const simulationBox::SimulationBox &, double, double);
    bool   applyRattle(double);

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    simulationBox::Molecule *getMolecule1() const { return _molecule1; }
    simulationBox::Molecule *getMolecule2() const { return _molecule2; }
    size_t                   getAtomIndex1() const { return _atomIndex1; }
    size_t                   getAtomIndex2() const { return _atomIndex2; }
    double                   getTargetBondLength() const { return _targetBondLength; }
    vector3d::Vec3D          getShakeDistanceRef() const { return _shakeDistanceRef; }
};

#endif   // _BOND_CONSTRAINT_HPP_