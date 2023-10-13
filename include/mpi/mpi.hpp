#ifndef _PIIMD_QMCF_MPI_HPP_

#define _PIIMD_QMCF_MPI_HPP_

#include <cstddef>   // for size_t

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

        static void setRank(const int &rank) { MPI::_rank = rank; }
        static void setProcessId(const int &processId) { MPI::_processId = processId; }

        [[nodiscard]] static int getRank() { return _rank; }
        [[nodiscard]] static int getProcessId() { return _processId; }
    };
}   // namespace mpi

#endif   // _PIIMD_QMCF_MPI_HPP_
