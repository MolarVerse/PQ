#include "../include/readInput/commandLineArgs.hpp"   // for CommandLineArgs
#include "engine.hpp"                                 // for Engine
#include "inputFileReader.hpp"                        // for InputFileReader
#include "mmmdEngine.hpp"                             // for MMMDEngine
#include "setup.hpp"                                  // for setupSimulation, setup

#include <cstdlib>     // for EXIT_SUCCESS
#include <exception>   // for exception
#include <iostream>    // for operator<<, basic_ostream, flush
#include <memory>      // for unique_ptr
#include <string>      // for string
#include <vector>      // for vector

#ifdef WITH_MPI
#include <mpi.h>
#endif

void test(std::unique_ptr<engine::Engine> &engine) { engine = std::unique_ptr<engine::MMMDEngine>(); }

static int pimdQmcf(int argc, const std::vector<std::string> &arguments)
{
    auto commandLineArgs = CommandLineArgs(argc, arguments);
    commandLineArgs.detectFlags();

    // auto  inputFileReader = readInput::InputFileReader(commandLineArgs.getInputFileName());
    // auto &engine         k ,= inputFileReader.readJobType();

    std::unique_ptr<engine::Engine> engine;

    test(engine);

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
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    }

#ifdef WITH_MPI
    for (int i = 1; i < size; i++)
    {
        auto path = "procId_pimd-qmcf_" + to_string(i);
        filesystem::remove_all(path);
    }
    MPI_Finalize();
#endif

    return EXIT_SUCCESS;
}