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

#include <cstdlib>      // for EXIT_FAILURE, EXIT_SUCCESS
#include <exception>    // for exception
#include <iostream>     // for operator<<
#include <memory>       // for unique_ptr
#include <string>       // for string, char_traits
#include <vector>       // for vector

#include "capabilities.hpp"      // for writeCapabilities
#include "commandLineArgs.hpp"   // for CommandLineArgs
#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for CustomException
#include "inputFileReader.hpp"   // for readJobType
#include "setup.hpp"             // for setupSimulation
#include "systemInfo.hpp"        // for _VERSION_

#ifdef WITH_MPI
#include <mpi.h>   // for MPI_Abort, MPI_COMM_WORLD, MPI_Finalize

#include "mpi.hpp"   // for MPI
#endif

#ifdef WITH_PYBIND11
#include <pybind11/embed.h>   // for scoped_interpreter
#endif

static int run(const std::string &inputFileName)
{
    auto engine = std::unique_ptr<engine::Engine>();
    input::readJobType(inputFileName, engine);

    setup::setupRequestedJob(inputFileName, *engine);

    /*
        HERE STARTS THE MAIN LOOP
    */

    engine->run();

    /*
        HERE ENDS THE MAIN LOOP
    */

    return EXIT_SUCCESS;
}

static void printHelp()
{
    std::cout << "Usage: PQ <input_file>\n"
              << "       PQ --help\n"
              << "       PQ --version\n"
              << "       PQ --capabilities=json\n\n"
              << "Run a PQ simulation from an input file.\n\n"
              << "Options:\n"
              << "  -h, --help     Show this help message.\n"
              << "  -V, --version  Show the PQ version.\n"
              << "  --capabilities=json\n"
              << "                  Show compiled capabilities as JSON.\n";
}

// main wrapper
int main(int argc, char *argv[])
{
    auto exitCode        = EXIT_SUCCESS;
    auto arguments       = std::vector<std::string>(argv, argv + argc);
    auto commandLineArgs = CommandLineArgs(argc, arguments);

    try
    {
        commandLineArgs.parse();
    }
    catch (const customException::CustomException &e)
    {
        std::cerr << "Error: " << e.getMessage() << '\n' << std::flush;
        return EXIT_FAILURE;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n' << std::flush;
        return EXIT_FAILURE;
    }

    if (CommandLineAction::HELP == commandLineArgs.getAction())
    {
        printHelp();
        return EXIT_SUCCESS;
    }

    if (CommandLineAction::VERSION == commandLineArgs.getAction())
    {
        std::cout << "PQ " << sysinfo::VERSION << '\n';
        return EXIT_SUCCESS;
    }

    if (CommandLineAction::CAPABILITIES == commandLineArgs.getAction())
    {
        cli::writeCapabilities(std::cout);
        return EXIT_SUCCESS;
    }

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
        exitCode = run(commandLineArgs.getInputFileName());
    }
    catch (const customException::CustomException &e)
    {
        std::cerr << "Error: " << e.getMessage() << '\n' << std::flush;
        exitCode = EXIT_FAILURE;

#ifdef WITH_MPI
        ::MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n' << std::flush;
        exitCode = EXIT_FAILURE;

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

    return exitCode;
}
