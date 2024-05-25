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

#ifdef WITH_KOKKOS
#include <Kokkos_Core.hpp>
#endif

class MyTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        #ifdef WITH_KOKKOS
        Kokkos::initialize();
        #endif
    }

    void TearDown() override {
        #ifdef WITH_KOKKOS
        Kokkos::finalize();
        #endif
    }
};

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Register our custom environment
    ::testing::AddGlobalTestEnvironment(new MyTestEnvironment);

    return RUN_ALL_TESTS();
}
