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

#ifndef _REFERENCES_OUTPUT_HPP_

#define _REFERENCES_OUTPUT_HPP_

#include <string>   // for string
#include <unordered_set>

#define REFERENCES_PATH_ _REFERENCES_PATH_

class ReferencesOutputTest;

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
        static inline std::unordered_set<std::string> _referenceFileNames;
        static inline std::unordered_set<std::string> _bibtexFileNames;

#ifdef WITH_TESTS
        friend class ::ReferencesOutputTest;
#endif

       public:
        static void writeReferencesFile();

        static void addReferenceFile(const std::string &referenceFileName);
    };

}   // namespace references

#endif   // _REFERENCES_OUTPUT_HPP_
