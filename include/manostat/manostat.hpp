#ifndef _MANOSTAT_HPP_

#define _MANOSTAT_HPP_

#include "vector3d.hpp"   // for Vec3D

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

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

      public:
        Manostat() = default;
        explicit Manostat(const double targetPressure) : _targetPressure(targetPressure) {}
        virtual ~Manostat() = default;

        void         calculatePressure(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
        virtual void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &);
    };

    /**
     * @class BerendsenManostat inherits from Manostat
     *
     * @link https://doi.org/10.1063/1.448118
     *
     */
    class BerendsenManostat : public Manostat
    {
      private:
        double _tau;
        double _compressibility;

      public:
        explicit BerendsenManostat(const double targetPressure, const double tau, const double compressibility)
            : Manostat(targetPressure), _tau(tau), _compressibility(compressibility){};

        void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] double getTau() const { return _tau; }
    };

}   // namespace manostat

#endif   // _MANOSTAT_HPP_