#ifndef _PIIMD_QMCF_MPI_HPP_

#define _PIIMD_QMCF_MPI_HPP_

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
        static inline int _processId;

        static void redirectOutput();

      public:
        static void init(int *argc, char ***argv);
        static void finalize();

        /********************
         * template methods *
         ********************/

        template <typename T>
        static void print(T t)
        {
            if (_rank != 0)
            {
                std::cout << t << std::endl;
            }
            else
            {
                std::cout << "RANK: " << _rank << "   " << t << std::endl;
            }
        }

        /***************************
         * standard setter methods *
         ***************************/

        static void setRank(const int &rank) { MPI::_rank = rank; }
        static void setProcessId(const int &processId) { MPI::_processId = processId; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static int getRank() { return _rank; }
        [[nodiscard]] static int getProcessId() { return _processId; }
    };
}   // namespace mpi

#endif   // _PIIMD_QMCF_MPI_HPP_
