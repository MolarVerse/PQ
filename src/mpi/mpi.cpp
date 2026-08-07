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

#include "mpi.hpp"

#include <mpi.h>   // for MPI_Comm_rank, MPI_Comm_size, MPI_Init, MPI_Finalize
#include <sys/stat.h>   // for mkdir
#include <unistd.h>     // for chdir

#include <filesystem>   // for remove_all
#include <format>       // for format
#include <fstream>      // for ofstream
#include <iostream>     // for cout, cerr

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

    const std::filesystem::path path = std::format("procId_PQ_{}", _rank);

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
        const auto path = std::format("procId_PQ_{}", i);
        std::filesystem::remove_all(path);
    }

    ::MPI_Finalize();
}

/***************************
 * standard setter methods *
 ***************************/

/**
 * @brief sets the rank
 *
 * @param rank
 */
void MPI::setRank(const size_t &rank) { MPI::_rank = rank; }

/**
 * @brief sets the size
 *
 * @param size
 */
void MPI::setSize(const size_t &size) { MPI::_size = size; }

/***************************
 * standard getter methods *
 ***************************/

/**
 * @brief check if rank is root
 *
 * @return true
 * @return false
 */
bool MPI::isRoot() { return _rank == 0; }

/**
 * @brief get rank
 *
 * @return size_t
 */
size_t MPI::getRank() { return _rank; }

/**
 * @brief get size
 *
 * @return size_t
 */
size_t MPI::getSize() { return _size; }