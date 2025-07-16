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

#include "randomNumberGenerator.hpp"

#include "settings.hpp"   // for Settings

using namespace randomNumberGenerator;
using namespace settings;

RandomNumberGenerator::RandomNumberGenerator()
{
    if (Settings::isRandomSeedSet())
        _generator.seed(Settings::getRandomSeed());
    else
        _generator.seed(_randomDevice());
}

/**
 * @brief get a random double from a normal distribution
 *
 * @param mean
 * @param stddev
 */
double RandomNumberGenerator::getNormalDistribution(double mean, double stddev)
{
    std::normal_distribution<double> distribution{mean, stddev};
    return distribution(_generator);
}

/**
 * @brief get a random double from a uniform real distribution
 *
 * @param min
 * @param max
 */
double RandomNumberGenerator::getUniformRealDistribution(double min, double max)
{
    std::uniform_real_distribution<double> distribution{min, max};
    return distribution(_generator);
}