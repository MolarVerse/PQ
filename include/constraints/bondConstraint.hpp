#ifndef _BOND_CONSTRAINT_HPP_

#define _BOND_CONSTRAINT_HPP_

#include "bond.hpp"
#include "molecule.hpp"
#include "simulationBox.hpp"

namespace constraints
{
    class BondConstraint;
}

/**
 * @class BondConstraint inherits from Bond
 *
 * @brief constraint object for single bond length
 *
 */
class constraints::BondConstraint : public connectivity::Bond
{
  private:
    double          _targetBondLength;
    vector3d::Vec3D _shakeDistanceRef;

  public:
    BondConstraint(simulationBox::Molecule *molecule1,
                   simulationBox::Molecule *molecule2,
                   size_t                   atomIndex1,
                   size_t                   atomIndex2,
                   double                   bondLength)
        : connectivity::Bond(molecule1, molecule2, atomIndex1, atomIndex2), _targetBondLength(bondLength){};

    void calculateConstraintBondRef(const simulationBox::SimulationBox &);

    double calculateDistanceDelta(const simulationBox::SimulationBox &) const;
    double calculateVelocityDelta() const;
    bool   applyShake(const simulationBox::SimulationBox &, double, double);
    bool   applyRattle(double);

    /**************************************
     *                                    *
     * standard getter and setter methods *
     *                                    *
     **************************************/

    void setShakeDistanceRef(vector3d::Vec3D shakeDistanceRef) { _shakeDistanceRef = shakeDistanceRef; }

    double          getTargetBondLength() const { return _targetBondLength; }
    vector3d::Vec3D getShakeDistanceRef() const { return _shakeDistanceRef; }
};

#endif   // _BOND_CONSTRAINT_HPP_