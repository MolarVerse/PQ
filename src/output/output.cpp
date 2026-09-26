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

#include "output.hpp"

#include <format>    // for format
#include <fstream>   // for ifstream, ofstream, std

#include "exceptions.hpp"           // for InputFileException, customException
#include "outputFileSettings.hpp"   // for OutputFileSettings

namespace out
{

    /**
     * @brief Sets the filename of the output file
     *
     * @param filename
     *
     * @throw InputFileException if filename is empty
     * @throw InputFileException if file already exists
     * and output should not be overwritten
     */
    void Output::setFilename(const std::string_view &filename)
    {
        _fileName = filename;
        const auto overwriteOutputFiles =
            settings::OutputFileSettings::getOverwriteOutputFiles();

        if (_fileName.empty())
            throw exc::InputFileException("Filename cannot be empty");

        if (const std::ifstream file(_fileName.c_str());
            file.good() && !overwriteOutputFiles)
            throw exc::InputFileException(
                "File already exists - filename = " + std::string(_fileName)
            );

        openFile();
    }

    /**
     * @brief Opens the output file
     *
     * @throw InputFileException if file cannot be opened
     *
     */
    void Output::openFile()
    {
        _fp.open(_fileName);

        if (!_fp.is_open())
            throw exc::InputFileException(
                "Could not open file - filename = " + _fileName
            );
    }

    /**
     * @brief Write the shared trajectory frame comment
     *
     * @param step simulation step
     */
    void Output::writeComment(size_t step)
    {
        if (settings::OutputFileSettings::getIncludeOutputMetadata())
            _fp << std::format("# step = {}\n", step);
        else
            _fp << '\n';
    }

    /**
     * @brief Write the shared trajectory force comment
     *
     * @param step simulation step
     * @param totalForce total force acting on the system
     */
    void Output::writeForceComment(size_t step, double totalForce)
    {
        _fp << formatForceComment(step, totalForce);
    }

    /**
     * @brief Formats the shared trajectory force comment
     *
     * @param step simulation step
     * @param totalForce total force acting on the system
     * @return formatted force comment
     */
    std::string Output::formatForceComment(
        const size_t step,
        const double totalForce
    )
    {
        const auto stepMetadata =
            settings::OutputFileSettings::getIncludeOutputMetadata()
                ? std::format("step = {}; ", step)
                : "";

        return format(
            "# {}Total force = {:.5e} kcal/mol/Angstrom\n",
            stepMetadata,
            totalForce
        );
    }

    /**
     * @brief Closes the output file
     *
     */
    void Output::close() { _fp.close(); }

    /**
     * @brief get filename
     *
     * @return string
     */
    std::string Output::getFilename() const { return _fileName; }

}   // namespace out
