#ifndef _BOND_CONSTRAINT_HPP_

#define _BOND_CONSTRAINT_HPP_

#include "bond.hpp"
#include "vector3d.hpp"

#include <cstddef>

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class Molecule;        // forward declaration
}   // namespace simulationBox

namespace constraints
{

    /**
     * @class BondConstraint inherits from Bond
     *
     * @brief constraint object for single bond length
     *
     * @details it performs the shake and rattle algorithm on a bond constraint
     *
     */
    class BondConstraint : public connectivity::Bond
    {
      private:
        double               _targetBondLength;
        linearAlgebra::Vec3D _shakeDistanceRef;

      public:
        BondConstraint(simulationBox::Molecule *molecule1,
                       simulationBox::Molecule *molecule2,
                       const size_t             atomIndex1,
                       const size_t             atomIndex2,
                       const double             bondLength)
            : connectivity::Bond(molecule1, molecule2, atomIndex1, atomIndex2), _targetBondLength(bondLength){};

        void calculateConstraintBondRef(const simulationBox::SimulationBox &);

        [[nodiscard]] double calculateDistanceDelta(const simulationBox::SimulationBox &) const;
        [[nodiscard]] double calculateVelocityDelta() const;

        [[nodiscard]] bool applyShake(const simulationBox::SimulationBox &, double tolerance);
        [[nodiscard]] bool applyRattle(double tolerance);

        /**************************************
         * standard getter and setter methods *
         **************************************/

        void setShakeDistanceRef(const linearAlgebra::Vec3D &shakeDistanceRef) { _shakeDistanceRef = shakeDistanceRef; }

        [[nodiscard]] double               getTargetBondLength() const { return _targetBondLength; }
        [[nodiscard]] linearAlgebra::Vec3D getShakeDistanceRef() const { return _shakeDistanceRef; }
    };

}   // namespace constraints

#endif   // _BOND_CONSTRAINT_HPP_