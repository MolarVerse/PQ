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

#ifndef _PARAMETER_FILE_READER_HPP_

#define _PARAMETER_FILE_READER_HPP_

#include <fstream>   // for ifstream
#include <memory>    // for unique_ptr
#include <string>
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "parameterFileSection.hpp"
#include "typeAliases.hpp"

namespace input::parameterFile
{
    void               readParameterFile(pq::Engine &);
    [[nodiscard]] bool isNeeded();

    /**
     * @class ParameterReader
     *
     * @brief reads parameter file and sets settings
     *
     */
    class ParameterFileReader
    {
       private:
        std::string   _fileName;
        std::ifstream _fp;
        pq::Engine   &_engine;

        pq::UniqueParamFileSectionVec _parameterFileSections;

       public:
        ParameterFileReader(const std::string &filename, pq::Engine &engine);

        void read();
        void deleteSection(const pq::ParamFileSection *section);

        [[nodiscard]] pq::ParamFileSection *determineSection(
            const pq::strings &lineElements
        );

        /**************************************
         * standard getter and setter methods *
         **************************************/

        void setFilename(const std::string_view &filename);

        [[nodiscard]] pq::UniqueParamFileSectionVec &getParameterFileSections();
        [[nodiscard]] const std::string             &getFilename() const;
    };

}   // namespace input::parameterFile

#endif   // _PARAMETER_FILE_READER_HPP_