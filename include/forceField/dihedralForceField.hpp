#ifndef _DIHEDRAL_FORCE_FIELD_HPP_

#define _DIHEDRAL_FORCE_FIELD_HPP_

#include "dihedral.hpp"

#include <cstddef>
#include <vector>

namespace potential
{
    class CoulombPotential;
    class NonCoulombPotential;
}   // namespace potential

namespace simulationBox
{
    class SimulationBox;
    class Molecule;
}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;
}

namespace forceField
{
    /**
     * @class DihedralForceField
     *
     * @brief Represents a dihedral between four atoms.
     *
     */
    class DihedralForceField : public connectivity::Dihedral
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
        void setForceConstant(const double forceConstant) { _forceConstant = forceConstant; }
        void setPeriodicity(const double periodicity) { _periodicity = periodicity; }
        void setPhaseShift(const double phaseShift) { _phaseShift = phaseShift; }

        [[nodiscard]] size_t getType() const { return _type; }
        [[nodiscard]] bool   isLinker() const { return _isLinker; }
        [[nodiscard]] double getForceConstant() const { return _forceConstant; }
        [[nodiscard]] double getPeriodicity() const { return _periodicity; }
        [[nodiscard]] double getPhaseShift() const { return _phaseShift; }
    };

}   // namespace forceField

#endif   // _DIHEDRAL_FORCE_FIELD_HPP_
