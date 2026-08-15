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

#include "driver.hpp"

#include "engine.hpp"
#include "inputFileReader.hpp"
#include "setup.hpp"

namespace driver
{
    /**
     * @brief Run a PQ simulation from an input file.
     *
     * @param inputFileName The name of the input file containing the simulation
     * setup.
     */
    void Driver::run(const std::string &inputFileName)
    {
        auto engine = std::unique_ptr<engine::Engine>();
        input::readJobType(inputFileName, engine);

        setup::setupRequestedJob(inputFileName, *engine);

        engine->run();
    }

}   // namespace driver
