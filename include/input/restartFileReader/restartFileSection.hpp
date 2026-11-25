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

#ifndef INPUT__RESTART_FILE_READER__RESTART_FILE_SECTION_HPP
#define INPUT__RESTART_FILE_READER__RESTART_FILE_SECTION_HPP

#include <fstream>
#include <string>
#include <vector>

struct Settings;   // forward declaration

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace input::restartFile
{
    /**
     * @class RestartFileSection
     *
     * @brief Base class for all sections of a .rst file
     *
     */
    class RestartFileSection
    {
       private:
        std::ifstream *_fp = nullptr;

       public:
        virtual ~RestartFileSection() = default;

        virtual std::string keyword()  = 0;
        virtual bool        isHeader() = 0;
        virtual void        process(
                   std::vector<std::string> &lineElements,
                   size_t                   &lineNumber,
                   Settings                 &settings,
                   molsys::SimulationBox    &simulationBox
               ) = 0;

        void setFilePointer(std::ifstream *file);

       protected:
        [[nodiscard]] std::ifstream *_getFilePointer() const;
    };

    inline void RestartFileSection::setFilePointer(std::ifstream *file)
    {
        _fp = file;
    }

    inline std::ifstream *RestartFileSection::_getFilePointer() const
    {
        return _fp;
    }

}   // namespace input::restartFile

#endif   // INPUT__RESTART_FILE_READER__RESTART_FILE_SECTION_HPP
