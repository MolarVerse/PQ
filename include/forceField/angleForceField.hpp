#ifndef _ANGLE_FORCE_FIELD_HPP_

#define _ANGLE_FORCE_FIELD_HPP_

#include "angle.hpp"

namespace forceField
{
    class AngleForceField;
}

/**
 * @class BondForceField inherits from Bond
 *
 * @brief force field object for single angle
 *
 */
class forceField::AngleForceField : public connectivity::Angle
{
  private:
    size_t _type;

  public:
    AngleForceField(const std::vector<simulationBox::Molecule *> &molecules, const std::vector<size_t> &atomIndices, size_t type)
        : connectivity::Angle(molecules, atomIndices), _type(type){};

    size_t getType() const { return _type; }
};

#endif   // _ANGLE_FORCE_FIELD_HPP_