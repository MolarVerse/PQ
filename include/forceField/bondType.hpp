#ifndef _BOND_TYPE_HPP_

#define _BOND_TYPE_HPP_

#include <cstddef>

namespace forceField
{
    class BondType;
}

/**
 * @class BondType
 *
 * @brief represents a bond type
 *
 */
class forceField::BondType
{
  private:
    size_t _id;

    double _equilibriumBondLength;
    double _forceConstant;

  public:
    BondType(size_t id, double equilibriumBondLength, double springConstant)
        : _id(id), _equilibriumBondLength(equilibriumBondLength), _forceConstant(springConstant){};

    [[nodiscard]] bool operator==(const BondType &other) const;

    [[nodiscard]] size_t getId() const { return _id; }
    [[nodiscard]] double getEquilibriumBondLength() const { return _equilibriumBondLength; }
    [[nodiscard]] double getForceConstant() const { return _forceConstant; }
};

#endif   // _BOND_TYPE_HPP_