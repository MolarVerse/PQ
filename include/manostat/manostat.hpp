#ifndef _MANOSTAT_HPP_

#define _MANOSTAT_HPP_

#include "vector3d.hpp"   // for Vec3D

namespace physicalData
{
    class PhysicalData;
}   // namespace physicalData

namespace simulationBox
{
    class SimulationBox;
}   // namespace simulationBox

namespace manostat
{
    /**
     * @class Manostat
     *
     * @brief Manostat is a base class for all manostats
     *
     */
    class Manostat
    {
      protected:
        linearAlgebra::Vec3D _pressureVector = {0.0, 0.0, 0.0};
        double               _pressure;
        double               _targetPressure;   // no default value, must be set

        double _timestep;

      public:
        Manostat() = default;
        explicit Manostat(const double targetPressure) : _targetPressure(targetPressure) {}
        virtual ~Manostat() = default;

        void         calculatePressure(physicalData::PhysicalData &physicalData);
        virtual void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &);

        /********************************
         * standard getters and setters *
         ********************************/

        void                 setTimestep(const double timestep) { _timestep = timestep; }
        [[nodiscard]] double getTimestep() const { return _timestep; }
    };

    /**
     * @class BerendsenManostat inherits from Manostat
     *
     */
    class BerendsenManostat : public Manostat
    {
      private:
        double _tau;
        double _compressibility;

      public:
        explicit BerendsenManostat(const double targetPressure, const double tau, const double compressibility)
            : Manostat(targetPressure), _tau(tau), _compressibility(compressibility)
        {
        }

        void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;

        [[nodiscard]] double getTau() const { return _tau; }
    };

}   // namespace manostat

#endif   // _MANOSTAT_HPP_