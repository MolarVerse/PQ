#include "mpi.hpp"

#include <filesystem>   // for remove_all
#include <format>       // for format
#include <fstream>      // for ofstream
#include <iostream>     // for cout, cerr
#include <mpi.h>        // for MPI_Comm_rank, MPI_Comm_size, MPI_Init, MPI_Finalize

using mpi::MPI;

/**
 * @brief Initializes MPI
 *
 * @param argc
 * @param argv
 */
void MPI::init(int *argc, char ***argv)
{
    ::MPI_Init(argc, argv);
    ::MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &_processId);
}

void MPI::redirectOutput()
{
    if (_rank != 0)
    {
        const std::ofstream sink("/dev/null");

        std::cout.rdbuf(sink.rdbuf());
        std::cerr.rdbuf(sink.rdbuf());
    }
}

/**
 * @brief Finalizes MPI
 *
 */
void MPI::finalize()
{
    for (int i = 1; i < _processId; ++i)
    {
        const auto path = std::format("procId_pimd-qmcf_{}", i);
        std::filesystem::remove_all(path);
    }

    ::MPI_Finalize();
}