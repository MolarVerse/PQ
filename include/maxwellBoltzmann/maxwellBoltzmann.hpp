#ifndef _MAXWELL_BOLTZMANN_HPP_

#define _MAXWELL_BOLTZMANN_HPP_

#include <random>

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace maxwellBoltzmann
{
    /**
     * @class MaxwellBoltzmann
     *
     * @brief class to initialize velocities of particles with a random maxwell boltzmann distribution
     *
     * @link https://www.biodiversitylibrary.org/item/53795#page/33/mode/1up
     * @link https://www.biodiversitylibrary.org/item/20012#page/37/mode/1up
     *
     */
    class MaxwellBoltzmann
    {
      private:
        std::random_device _randomDevice{};
        std::mt19937       _generator{_randomDevice()};

      public:
        void initializeVelocities(simulationBox::SimulationBox &);
    };
}   // namespace maxwellBoltzmann

#endif   // _MAXWELL_BOLTZMANN_HPP_