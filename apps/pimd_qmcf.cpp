/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "commandLineArgs.hpp"   // for CommandLineArgs
#include "engine.hpp"            // for Engine
#include "inputFileReader.hpp"   // for readJobType
#include "mpi.hpp"               // for MPI
#include "setup.hpp"             // for setupSimulation

#include <cstdlib>      // for EXIT_SUCCESS
#include <exception>    // for exception
#include <filesystem>   // for remove_all
#include <iostream>     // for operator<<
#include <memory>       // for unique_ptr
#include <string>       // for string, char_traits
#include <vector>       // for vector

#ifdef WITH_MPI
    #include <mpi.h>
#endif

static int pimdQmcf(int argc, const std::vector<std::string> &arguments)
{
    auto commandLineArgs = CommandLineArgs(argc, arguments);
    commandLineArgs.detectFlags();

    auto engine = std::unique_ptr<engine::Engine>();
    input::readJobType(commandLineArgs.getInputFileName(), engine);

    setup::setupSimulation(commandLineArgs.getInputFileName(), *engine);

    /*
        HERE STARTS THE MAIN LOOP
    */

    engine->run();

    /*
        HERE ENDS THE MAIN LOOP
    */

    return EXIT_SUCCESS;
}

// main wrapper
int main(int argc, char *argv[])
{
#ifdef WITH_MPI
    mpi::MPI::init(&argc, &argv);
#endif

    try
    {
        auto arguments = std::vector<std::string>(argv, argv + argc);
        ::pimdQmcf(argc, arguments);
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception: " << e.what() << '\n' << std::flush;

#ifdef WITH_MPI
        ::MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    }

#ifdef WITH_MPI
    mpi::MPI::finalize();
#endif

    return EXIT_SUCCESS;
}