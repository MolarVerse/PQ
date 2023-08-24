#ifndef _DIHEDRAL_FORCE_FIELD_HPP_

#define _DIHEDRAL_FORCE_FIELD_HPP_

#include "coulombPotential.hpp"
#include "dihedral.hpp"
#include "molecule.hpp"
#include "nonCoulombPotential.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

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
    bool   _isLinker = false;

    double _forceConstant;
    double _periodicity;
    double _phaseShift;

  public:
    DihedralForceField(const std::vector<simulationBox::Molecule *> &molecules,
                       const std::vector<size_t>                    &atomIndices,
                       size_t                                        type)
        : connectivity::Dihedral(molecules, atomIndices), _type(type){};

    void calculateEnergyAndForces(const simulationBox::SimulationBox &,
                                  physicalData::PhysicalData &,
                                  bool,
                                  const potential::CoulombPotential &,
                                  potential::NonCoulombPotential &);

    void setIsLinker(const bool isLinker) { _isLinker = isLinker; }
    void setForceConstant(double forceConstant) { _forceConstant = forceConstant; }
    void setPeriodicity(double periodicity) { _periodicity = periodicity; }
    void setPhaseShift(double phaseShift) { _phaseShift = phaseShift; }

    [[nodiscard]] size_t getType() const { return _type; }
    [[nodiscard]] double getForceConstant() const { return _forceConstant; }
    [[nodiscard]] double getPeriodicity() const { return _periodicity; }
    [[nodiscard]] double getPhaseShift() const { return _phaseShift; }
};

#endif   // _DIHEDRAL_FORCE_FIELD_HPP_
