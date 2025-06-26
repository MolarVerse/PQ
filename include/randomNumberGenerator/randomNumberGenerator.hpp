/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _RANDOM_NUMBER_GENERATOR_HPP_

#define _RANDOM_NUMBER_GENERATOR_HPP_

#include <random>   // for random_device and mt19937

namespace randomNumberGenerator
{
    /**
     * @class RandomNumberGenerator
     *
     * @brief class to handle generation of random numbers
     *
     */
    class RandomNumberGenerator
    {
       private:
        std::random_device _randomDevice{};
        std::mt19937       _generator;

       public:
        RandomNumberGenerator();
        double getNormalDistribution(double mean, double stddev);
    };

}   // namespace randomNumberGenerator

#endif   // _RANDOM_NUMBER_GENERATOR_HPP_