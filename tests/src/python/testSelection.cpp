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

#ifdef WITH_WITH_PYTHON_BINDINGS

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <vector>

#include "selection.hpp"

TEST(TestPythonSelection, selectsAtomsFromRestartFile)
{
    const auto restartFile =
        std::filesystem::temp_directory_path() / "pqSelection.rst";

    {
        std::ofstream file{restartFile};
        file << "Box 10.0 10.0 10.0\n";
        file << "O 1 1 0.0 0.0 0.0\n";
        file << "H 2 1 1.0 0.0 0.0\n";
        file << "H 3 1 0.0 1.0 0.0\n";
        file << "Na 4 2 5.0 5.0 5.0\n";
    }

    EXPECT_EQ(
        pq_python::select("O", restartFile.string(), ""),
        (std::vector<int>{0})
    );
    EXPECT_EQ(
        pq_python::select("elem(H), Na", restartFile.string(), ""),
        (std::vector<int>{1, 2, 3})
    );

    std::filesystem::remove(restartFile);
}

#endif   // WITH_WITH_PYTHON_BINDINGS
