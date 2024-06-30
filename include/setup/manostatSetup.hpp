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

#ifndef _MANOSTAT_SETUP_HPP_

#define _MANOSTAT_SETUP_HPP_

namespace engine
{
    class Engine;     // forward declaration
    class MDEngine;   // forward declaration
}   // namespace engine

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
        explicit ManostatSetup(engine::MDEngine &engine) : _engine(engine){};

        void setup();

        void isPressureSet() const;
        void setupBerendsenManostat();
        void setupStochasticRescalingManostat();

        [[nodiscard]] engine::MDEngine &getEngine() const { return _engine; }
    };

}   // namespace setup

#endif   // _MANOSTAT_SETUP_HPP_