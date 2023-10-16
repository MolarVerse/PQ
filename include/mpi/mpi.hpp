#ifdef WITH_MPI

#ifndef _PIMD_QMCF_MPI_HPP_

#define _PIMD_QMCF_MPI_HPP_

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
        static inline int _rank = 0;
        static inline int _size;

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

        static void setRank(const int &rank) { MPI::_rank = rank; }
        static void setSize(const int &size) { MPI::_size = size; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static bool isRoot() { return _rank == 0; }
        [[nodiscard]] static int  getRank() { return _rank; }
        [[nodiscard]] static int  getSize() { return _size; }
    };
}   // namespace mpi

#endif   // _PIMD_QMCF_MPI_HPP_

#endif   // WITH_MPI