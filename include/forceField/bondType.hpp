#ifndef _BOND_TYPE_HPP_

#define _BOND_TYPE_HPP_

#include <cstddef>

namespace forceField
{
    /**
     * @class BondType
     *
     * @brief represents a bond type
     *
     * @details this is a class representing a bond type defined in the parameter file
     *
     */
    class BondType
    {
      private:
        size_t _id;

        double _equilibriumBondLength;
        double _forceConstant;

      public:
        BondType(size_t id, double equilibriumBondLength, double springConstant)
            : _id(id), _equilibriumBondLength(equilibriumBondLength), _forceConstant(springConstant){};

        [[nodiscard]] bool operator==(const BondType &other) const;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getId() const { return _id; }
        [[nodiscard]] double getEquilibriumBondLength() const { return _equilibriumBondLength; }
        [[nodiscard]] double getForceConstant() const { return _forceConstant; }
    };

}   // namespace forceField

#endif   // _BOND_TYPE_HPP_