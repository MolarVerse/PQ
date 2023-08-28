#include <filesystem>
#include <iostream>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "celllist.hpp"
#include "commandLineArgs.hpp"
#include "engine.hpp"
#include "setup.hpp"

using namespace std;
using namespace setup;
using namespace engine;

static int pimdQmcf(int argc, const std::vector<std::string> &arguments)
{
    auto commandLineArgs = CommandLineArgs(argc, arguments);
    commandLineArgs.detectFlags();

    auto engine = Engine();

    setupSimulation(commandLineArgs.getInputFileName(), engine);

    /*
        HERE STARTS THE MAIN LOOP
    */

    engine.run();

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
        auto arguments = vector<string>(argv, argv + argc);
        ::pimdQmcf(argc, arguments);
    }
    catch (const exception &e)
    {
        cout << "Exception: " << e.what() << '\n' << flush;
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