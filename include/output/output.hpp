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

#include <cstddef>       // for size_t
#include <fstream>       // for ofstream
#include <string>        // for string
#include <string_view>   // for string_view
#include <utility>

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

        void               openFile();
        void               writeComment(std::size_t);
        void               writeForceComment(std::size_t, double);
        static std::string formatForceComment(std::size_t, double);

       public:
        explicit Output(std::string filename) : _fileName(std::move(filename))
        {
        }
        ~Output() { close(); }
        Output(const Output &)                = delete;
        Output &operator=(const Output &)     = delete;
        Output(Output &&) noexcept            = default;
        Output &operator=(Output &&) noexcept = default;

        void setFilename(const std::string_view &);
        void close();

#ifdef WITH_TESTS
        friend class ::TestOutput_testSpecialSetFilename_Test;
#endif

        /***************************
         * standard getter methods *
         ***************************/

        std::string getFilename() const;
    };

}   // namespace output

#endif   // _OUTPUT_HPP_
