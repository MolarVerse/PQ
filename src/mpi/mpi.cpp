#include "mpi.hpp"

#include <filesystem>   // for remove_all
#include <format>       // for format
#include <fstream>      // for ofstream
#include <iostream>     // for cout, cerr
#include <mpi.h>        // for MPI_Comm_rank, MPI_Comm_size, MPI_Init, MPI_Finalize
#include <sys/stat.h>   // for mkdir
#include <unistd.h>     // for chdir

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

    int rank;
    int size;

    ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &size);

    _rank = size_t(rank);
    _size = size_t(size);

    setupMPIDirectories();

    redirectOutput();

    ::MPI_Barrier(MPI_COMM_WORLD);
}

void MPI::setupMPIDirectories()
{
    if (_rank == 0)
        return;

    const std::filesystem::path path = std::format("procId_pimd-qmcf_{}", _rank);

    std::filesystem::remove_all(path.c_str());
    std::filesystem::create_directory(path);
    for (const auto &entry : std::filesystem::directory_iterator("."))
    {
        if (entry.is_directory())
            continue;

        std::filesystem::copy(entry.path(), path);
    }
    std::filesystem::current_path(path);
}

/**
 * @brief Redirects stdout output to /dev/null for all ranks except rank 0
 *
 */
void MPI::redirectOutput()
{
    if (_rank != 0)
        std::cout.rdbuf(nullptr);
}

/**
 * @brief Finalizes MPI
 *
 */
void MPI::finalize()
{
    for (size_t i = 1; i < _size; ++i)
    {
        const auto path = std::format("procId_pimd-qmcf_{}", i);
        std::filesystem::remove_all(path);
    }

    ::MPI_Finalize();
}