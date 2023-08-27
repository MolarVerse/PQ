#ifndef _ANGLE_FORCE_FIELD_HPP_

#define _ANGLE_FORCE_FIELD_HPP_

#include "angle.hpp"

#include <cstddef>   // for size_t
#include <vector>    // for vector

namespace simulationBox
{
    class SimulationBox;
    class Molecule;
}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;
}   // namespace physicalData

namespace potential
{
    class CoulombPotential;
    class NonCoulombPotential;
}   // namespace potential

namespace forceField
{
    /**
     * @class BondForceField inherits from Bond
     *
     * @brief force field object for single angle
     *
     */
    class AngleForceField : public connectivity::Angle
    {
      private:
        size_t _type;
        bool   _isLinker = false;

        double _equilibriumAngle;
        double _forceConstant;

      public:
        AngleForceField(const std::vector<simulationBox::Molecule *> &molecules,
                        const std::vector<size_t>                    &atomIndices,
                        size_t                                        type)
            : connectivity::Angle(molecules, atomIndices), _type(type){};

        void calculateEnergyAndForces(const simulationBox::SimulationBox &,
                                      physicalData::PhysicalData &,
                                      const potential::CoulombPotential &,
                                      potential::NonCoulombPotential &);

        void setIsLinker(const bool isLinker) { _isLinker = isLinker; }
        void setEquilibriumAngle(const double equilibriumAngle) { _equilibriumAngle = equilibriumAngle; }
        void setForceConstant(const double forceConstant) { _forceConstant = forceConstant; }

        [[nodiscard]] size_t getType() const { return _type; }
        [[nodiscard]] bool   isLinker() const { return _isLinker; }
        [[nodiscard]] double getEquilibriumAngle() const { return _equilibriumAngle; }
        [[nodiscard]] double getForceConstant() const { return _forceConstant; }
    };

}   // namespace forceField

#endif   // _ANGLE_FORCE_FIELD_HPP_