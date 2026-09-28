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

#include <gtest/gtest.h>

#ifdef WITH_MPI
#include "mpi.hpp"   // for MPI
#endif

int main(int argc, char **argv)
{
#ifdef WITH_MPI
    // code under test may call MPI collectives (e.g. MPI_Bcast), which
    // require MPI to be initialized (single rank when run via ctest)
    mpi::MPI::init(&argc, &argv);
#endif

    ::testing::InitGoogleTest(&argc, argv);
    const auto result = RUN_ALL_TESTS();

#ifdef WITH_MPI
    mpi::MPI::finalize();
#endif

    return result;
}
