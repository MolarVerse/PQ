#ifndef _DIHEDRAL_FORCE_FIELD_HPP_

#define _DIHEDRAL_FORCE_FIELD_HPP_

#include "dihedral.hpp"
#include "molecule.hpp"

#include <cstddef>
#include <vector>

namespace forceField
{
    class DihedralForceField;
}   // namespace forceField

/**
 * @class DihedralForceField
 *
 * @brief Represents a dihedral between four atoms.
 *
 */
class forceField::DihedralForceField : public connectivity::Dihedral
{
  private:
    size_t _type;

    double _forceConstant;
    double _periodicity;
    double _phaseShift;

  public:
    DihedralForceField(const std::vector<simulationBox::Molecule *> &molecules,
                       const std::vector<size_t>                    &atomIndices,
                       size_t                                        type)
        : connectivity::Dihedral(molecules, atomIndices), _type(type){};

    void setForceConstant(double forceConstant) { _forceConstant = forceConstant; }
    void setPeriodicity(double periodicity) { _periodicity = periodicity; }
    void setPhaseShift(double phaseShift) { _phaseShift = phaseShift; }

    size_t getType() const { return _type; }
    double getForceConstant() const { return _forceConstant; }
    double getPeriodicity() const { return _periodicity; }
    double getPhaseShift() const { return _phaseShift; }
};

#endif   // _DIHEDRAL_FORCE_FIELD_HPP_
