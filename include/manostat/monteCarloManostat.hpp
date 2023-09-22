#ifndef _MONTE_CARLO_MANOSTAT_HPP_

#define _MONTE_CARLO_MANOSTAT_HPP_

#include "manostat.hpp"   // for Manostat

#include <random>   // for std::random_device, std::mt19937

namespace manostat
{
    class MonteCarloManostat : public Manostat
    {
      private:
        std::random_device _randomDevice{};
        std::mt19937       _generator{_randomDevice()};

      public:
        explicit MonteCarloManostat() = default;

        void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override{};
    };
}   // namespace manostat

#endif   // _MONTE_CARLO_MANOSTAT_HPP_
