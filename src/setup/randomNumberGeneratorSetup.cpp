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

#include "randomNumberGeneratorSetup.hpp"

#include "engine.hpp"            // for Engine
#include "settings.hpp"          // for Settings
#include "stringUtilities.hpp"   // for toLowerCopy

using ::setup::RandomNumberGeneratorSetup;
using namespace settings;
using namespace engine;

/**
 * @brief wrapper to build randomNumberGeneratorSetup object and call setup
 *
 * @param engine
 */
void setup::setupRandomNumberGenerator(Engine &engine)
{
    engine.getLogOutput().writeSetup("Random Number Generator");

    RandomNumberGeneratorSetup randomNumberGeneratorSetup(engine);
    randomNumberGeneratorSetup.setup();
}

/**
 * @brief constructor
 *
 * @param engine
 */
RandomNumberGeneratorSetup::RandomNumberGeneratorSetup(Engine &engine)
    : _engine(engine)
{
}

/**
 * @brief setup the random number generator
 *
 */
void RandomNumberGeneratorSetup::setup() { setupWriteInfo(); }

/**
 * @brief write info about the random number generator setup
 *
 */
void RandomNumberGeneratorSetup::setupWriteInfo() const
{
    auto &logOutput = _engine.getLogOutput();

    if (Settings::isRandomSeedSet())
    {
        const auto randomSeed = Settings::getRandomSeed();
        const auto randomNumberGeneratorMessage =
            std::format("Random seed has been set to: {}", randomSeed);

        logOutput.writeSetupInfo(randomNumberGeneratorMessage);
    }

    else
    {
        const auto randomNumberGeneratorMessage =
            std::format("Using system-generated random seed");

        logOutput.writeSetupInfo(randomNumberGeneratorMessage);
    }

    logOutput.writeEmptyLine();
}