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

#include <cstdlib>     // for EXIT_FAILURE, EXIT_SUCCESS
#include <exception>   // for exception
#include <iostream>    // for operator<<
#include <string>      // for string, char_traits
#include <vector>      // for vector

#include "capabilities.hpp"      // for writeCapabilities
#include "commandLineArgs.hpp"   // for CommandLineArgs
#include "driver.hpp"
#include "exceptions.hpp"   // for CustomException
#include "systemInfo.hpp"   // for _VERSION_
#include "validation.hpp"   // for validation

#ifdef WITH_MPI
#include <mpi.h>   // for MPI_Abort, MPI_COMM_WORLD, MPI_Finalize

#include "mpi.hpp"   // for MPI
#endif

#ifdef WITH_PYBIND11
#include <pybind11/embed.h>   // for scoped_interpreter
#endif
namespace
{
    void printHelp()
    {
        std::cout
            << "Usage: PQ <input_file>\n"
            << "       PQ --help\n"
            << "       PQ --version\n"
            << "       PQ --capabilities=json\n"
            << "       PQ --validate <input_file> [--format=text|json] "
               "[--scope=installed|portable]\n\n"
            << "Run a PQ simulation from an input file.\n\n"
            << "Options:\n"
            << "  -h, --help       Show this help message.\n"
            << "  -V, --version    Show the PQ version.\n"
            << "  --capabilities=json\n"
            << "                    Show compiled capabilities as JSON.\n"
            << "  --validate <input_file>\n"
            << "                    Check input without running a simulation.\n"
            << "  --format=text     Return readable validation (default).\n"
            << "  --format=json     Return machine-readable validation.\n"
            << "  --scope=installed Check this build and referenced files "
               "(default).\n"
            << "  --scope=portable  Check portable input semantics only.\n";
    }
}   // namespace

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

    if (CommandLineAction::VALIDATE == commandLineArgs.getAction())
    {
        try
        {
            const auto result = cli::validateInputFile(
                commandLineArgs.getInputFileName(),
                commandLineArgs.getValidationScope()
            );

            if (CommandLineFormat::JSON == commandLineArgs.getFormat())
                cli::writeValidationJson(result, std::cout);
            else
                cli::writeValidationText(result, std::cout, std::cerr);

            return result.valid ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Validation failed: " << e.what() << '\n'
                      << std::flush;
            return 2;
        }
    }

#ifdef WITH_MPI
    mpi::MPI::init(&argc, &argv);
#endif

#ifdef WITH_PYBIND11
    pybind11::scoped_interpreter guard{};
#endif

    try
    {
        driver::Driver().run(commandLineArgs.getInputFileName());
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

#ifdef WITH_MPI
    mpi::MPI::finalize();
#endif

    return exitCode;
}
