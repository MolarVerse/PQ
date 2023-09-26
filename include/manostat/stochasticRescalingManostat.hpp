#ifndef _STOCHASTIC_RESCALING_MANOSTAT_HPP_

#define _STOCHASTIC_RESCALING_MANOSTAT_HPP_

#include "manostat.hpp"   // for Manostat

#include <random>   // for std::random_device, std::mt19937

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
     * @class StochasticRescalingManostat inherits from Manostat
     *
     * @link https://doi.org/10.1063/5.0020514
     *
     */
    class StochasticRescalingManostat : public Manostat
    {
      private:
        std::random_device _randomDevice{};
        std::mt19937       _generator{_randomDevice()};

        double _tau;
        double _compressibility;

      public:
        StochasticRescalingManostat() = default;
        StochasticRescalingManostat(const StochasticRescalingManostat &);
        explicit StochasticRescalingManostat(const double targetPressure, const double tau, const double compressibility)
            : Manostat(targetPressure), _tau(tau), _compressibility(compressibility){};

        void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] double getTau() const { return _tau; }
    };

}   // namespace manostat

#endif   // _STOCHASTIC_RESCALING_MANOSTAT_HPP_