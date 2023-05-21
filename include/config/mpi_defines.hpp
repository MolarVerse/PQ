#ifndef _MPI_DEFINES_

#define _MPI_FINALIZE_ \
    #ifdef WITH_MPI    \
    MPI_finalize();    \
    #endif

#endif // _MPI_DEFINES_
