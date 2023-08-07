#ifndef _DIHEDRAL_TYPE_HPP_

#define _DIHEDRAL_TYPE_HPP_

#include "cstddef"

namespace forceField
{
    class DihedralType;
}

class forceField::DihedralType
{
  private:
    size_t _id;

    double _forceConstant;
    double _periodicity;
    double _phaseShift;

  public:
    DihedralType(size_t id, double forceConstant, double frequency, double phaseShift)
        : _id(id), _forceConstant(forceConstant), _periodicity(frequency), _phaseShift(phaseShift){};

    bool operator==(const DihedralType &other) const;

    size_t getId() const { return _id; }
    double getForceConstant() const { return _forceConstant; }
    double getPeriodicity() const { return _periodicity; }
    double getPhaseShift() const { return _phaseShift; }
};

#endif   // _DIHEDRAL_TYPE_HPP_