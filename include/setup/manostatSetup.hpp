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

#ifndef SETUP__MANOSTAT_SETUP_HPP
#define SETUP__MANOSTAT_SETUP_HPP

namespace engine
{
    class Engine;     // forward declaration
    class MDEngine;   // forward declaration
}   // namespace engine

class ManostatSettings;   // forward declaration

namespace out
{
    class LogOutput;   // forward declaration
}   // namespace out

namespace setup
{
    void setupManostat(engine::Engine &);

    /**
     * @class ManostatSetup
     *
     * @brief Setup manostat
     *
     */
    class ManostatSetup
    {
       private:
        engine::MDEngine &_engine;

       public:
        explicit ManostatSetup(engine::MDEngine &engine);

        void setup();

        void validateIsotropyFixedAxisCombination() const;

        [[nodiscard]] engine::MDEngine &getEngine() const;

       protected:
        static void _writeManostatSelection(
            out::LogOutput         &logOutput,
            const ManostatSettings &manostatSettings
        );
        static void _writeBerendsenSetup(
            out::LogOutput         &logOutput,
            const ManostatSettings &settings
        );
        static void _writeIsotropy(
            out::LogOutput         &logOutput,
            const ManostatSettings &settings
        );

       private:
        void _writeSetupInfo() const;
    };

}   // namespace setup

#endif   // SETUP__MANOSTAT_SETUP_HPP
