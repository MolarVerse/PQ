/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#ifndef _REFERENCES_OUTPUT_HPP_

#define _REFERENCES_OUTPUT_HPP_

#define REFERENCES_PATH_ _REFERENCES_PATH_

#include <set>      // for set
#include <string>   // for string

namespace references
{
    /**
     * @class ReferencesOutput
     *
     * @brief class to print references file
     *
     */
    class ReferencesOutput
    {
      private:
        static inline std::string _referenceFilesPath = REFERENCES_PATH_;

        static inline std::set<std::string> _referenceFileNames = std::set<std::string>();
        static inline std::set<std::string> _bibtexFileNames    = std::set<std::string>();

      public:
        static void writeReferencesFile();

        static void addReferenceFile(const std::string &referenceFileName);
    };

}   // namespace references

#endif   // _REFERENCES_OUTPUT_HPP_