#ifndef _ANGLE_FORCE_FIELD_HPP_

#define _ANGLE_FORCE_FIELD_HPP_

#include "angle.hpp"
#include "molecule.hpp"

#include <cstddef>
#include <vector>

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

    double _equilibriumAngle;
    double _forceConstant;

  public:
    AngleForceField(const std::vector<simulationBox::Molecule *> &molecules, const std::vector<size_t> &atomIndices, size_t type)
        : connectivity::Angle(molecules, atomIndices), _type(type){};

    void setEquilibriumAngle(double equilibriumAngle) { _equilibriumAngle = equilibriumAngle; }
    void setForceConstant(double forceConstant) { _forceConstant = forceConstant; }

    [[nodiscard]] size_t getType() const { return _type; }
    [[nodiscard]] double getEquilibriumAngle() const { return _equilibriumAngle; }
    [[nodiscard]] double getForceConstant() const { return _forceConstant; }
};

#endif   // _ANGLE_FORCE_FIELD_HPP_