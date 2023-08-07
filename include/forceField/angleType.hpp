#ifndef _ANGLE_TYPE_HPP_

#define _ANGLE_TYPE_HPP_

#include "cstddef"

namespace forceField
{
    class AngleType;
}

class forceField::AngleType
{
  private:
    size_t _id;

    double _equilibriumAngle;
    double _forceConstant;

  public:
    AngleType(size_t id, double equilibriumAngle, double springConstant)
        : _id(id), _equilibriumAngle(equilibriumAngle), _forceConstant(springConstant){};

    bool operator==(const AngleType &other) const;

    size_t getId() const { return _id; }
    double getEquilibriumAngle() const { return _equilibriumAngle; }
    double getForceConstant() const { return _forceConstant; }
};

#endif   // _ANGLE_TYPE_HPP_