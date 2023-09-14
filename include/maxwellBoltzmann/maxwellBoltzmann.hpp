#ifndef _MAXWELL_BOLTZMANN_HPP_

#define _MAXWELL_BOLTZMANN_HPP_

#include "simulationBox.hpp"

#include <random>

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
        std::default_random_engine       generator;
        std::normal_distribution<double> distribution{0.5, 1.0};
    };
}   // namespace maxwellBoltzmann

#endif   // _MAXWELL_BOLTZMANN_HPP_