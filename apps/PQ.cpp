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

#include <cstdlib>      // for EXIT_SUCCESS
#include <exception>    // for exception
#include <filesystem>   // for remove_all
#include <iostream>     // for operator<<
#include <memory>       // for unique_ptr
#include <string>       // for string, char_traits
#include <vector>       // for vector

#include "commandLineArgs.hpp"   // for CommandLineArgs
#include "engine.hpp"            // for Engine
#include "inputFileReader.hpp"   // for readJobType
#include "setup.hpp"             // for setupSimulation

#ifdef WITH_MPI
#include <mpi.h>   // for MPI_Abort, MPI_COMM_WORLD, MPI_Finalize

#include "mpi.hpp"   // for MPI
#endif

#ifdef WITH_PYBIND11
#include <pybind11/embed.h>   // for scoped_interpreter
#endif

static int PQ(int argc, const std::vector<std::string> &arguments)
{
    auto commandLineArgs = CommandLineArgs(argc, arguments);
    commandLineArgs.detectFlags();

    auto engine = std::unique_ptr<engine::Engine>();
    input::readJobType(commandLineArgs.getInputFileName(), engine);

    setup::setupRequestedJob(commandLineArgs.getInputFileName(), *engine);

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

#ifdef WITH_KOKKOS
    Kokkos::initialize(argc, argv);
#endif

#ifdef WITH_PYBIND11
    pybind11::scoped_interpreter guard{};
#endif

    try
    {
        auto arguments = std::vector<std::string>(argv, argv + argc);
        ::PQ(argc, arguments);
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception: " << e.what() << '\n' << std::flush;

#ifdef WITH_MPI
        ::MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    }

#ifdef WITH_KOKKOS
    Kokkos::finalize();
#endif

#ifdef WITH_MPI
    mpi::MPI::finalize();
#endif

    return EXIT_SUCCESS;
}
