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

#ifndef _OUTPUT_HPP_

#define _OUTPUT_HPP_

#include <fstream>       // for ofstream
#include <string>        // for string
#include <string_view>   // for string_view

#ifdef WITH_TESTS
#include <gtest/gtest_prod.h>   // for FRIEND_TEST
#endif

class TestOutput_testSpecialSetFilename_Test;   // Friend test class

namespace output
{
    /**
     * @class Output
     *
     * @brief Base class for output files
     *
     */
    class Output
    {
       protected:
        std::string   _fileName;
        std::ofstream _fp;
        int           _rank;

        void openFile();

       public:
        explicit Output(const std::string &filename) : _fileName(filename){};

        void setFilename(const std::string_view &);
        void close() { _fp.close(); }

#ifdef WITH_TESTS
        FRIEND_TEST(::TestOutput, testSpecialSetFilename);
#endif

        /********************************
         * standard getters and setters *
         ********************************/

        std::string getFilename() const { return _fileName; }
    };

}   // namespace output

#endif   // _OUTPUT_HPP_