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

#ifdef WITH_MPI

#ifndef _PQ_MPI_HPP_

#define _PQ_MPI_HPP_

#include <cstddef>    // for size_t
#include <fstream>    // for ofstream
#include <iostream>   // for cout, cerr

namespace mpi
{
    /**
     * @class MPI
     *
     * @brief Wrapper for MPI
     *
     */
    class MPI
    {
       private:
        static inline size_t _rank = 0;
        static inline size_t _size;

        static void setupMPIDirectories();
        static void redirectOutput();

       public:
        static void init(int *argc, char ***argv);
        static void finalize();

        /********************
         * template methods *
         ********************/

        /**
         * @brief prints to stderr for all ranks
         *
         * @tparam T
         * @param t
         */
        template <typename T>
        static void print(T t)
        {
            std::cerr << "RANK: " << _rank << "   " << t << std::endl;
        }

        /***************************
         * standard setter methods *
         ***************************/

        static void setRank(const size_t &rank);
        static void setSize(const size_t &size);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static bool   isRoot();
        [[nodiscard]] static size_t getRank();
        [[nodiscard]] static size_t getSize();
    };
}   // namespace mpi

#endif   // _PQ_MPI_HPP_

#endif   // WITH_MPI