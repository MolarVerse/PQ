#ifndef _CONSTRAINTS_HPP_

#define _CONSTRAINTS_HPP_

#include "bondConstraint.hpp"

/**
 * @brief namespace for all constraints
 */
namespace constraints
{
    class Constraints;
}

/**
 * @class Constraints
 */
class constraints::Constraints
{
  private:
    bool _activated = false;

    std::vector<BondConstraint> _bondConstraints;

  public:
    void activate() { _activated = true; }
    bool isActivated() const { return _activated; }

    void addBondConstraint(BondConstraint bondConstraint) { _bondConstraints.push_back(bondConstraint); }

    const std::vector<BondConstraint> &getBondConstraints() const { return _bondConstraints; }
};

#endif   // _CONSTRAINTS_HPP_