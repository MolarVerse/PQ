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

#ifndef _CONSTRAINTS_SETUP_HPP_

#define _CONSTRAINTS_SETUP_HPP_

#include "typeAliases.hpp"

namespace setup
{
    void setupConstraints(pq::Engine &);

    /**
     * @class ConstraintsSetup
     *
     * @brief Setup constraints before reading guffdat file
     *
     */
    class ConstraintsSetup
    {
       private:
        pq::Engine &_engine;

        size_t _shakeConstraints  = 0;
        size_t _mShakeConstraints = 0;

        size_t _shakeMaxIter  = 0;
        size_t _rattleMaxIter = 0;

        double _shakeTolerance  = 0.0;
        double _rattleTolerance = 0.0;

       public:
        explicit ConstraintsSetup(pq::Engine &engine);

        void setup();

        void setupMShake();

        void setupTolerances();
        void setupMaxIterations();
        void setupRefBondLengths();
        void setupDegreesOfFreedom();

        void writeSetupInfo();
        void writeEnabled();
        void writeDof();
        void writeTolerance();
        void writeMaxIter();
        void writeNConstraintBonds();
    };

}   // namespace setup

#endif   // _CONSTRAINTS_SETUP_HPP_