#ifndef _BERENDSEN_MANOSTAT_HPP_

#define _BERENDSEN_MANOSTAT_HPP_

#include "manostat.hpp"   // for Manostat

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace manostat
{
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

#endif   // _BERENDSEN_MANOSTAT_HPP_