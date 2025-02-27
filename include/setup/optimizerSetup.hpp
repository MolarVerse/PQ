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

#ifndef _OPTIMIZER_SETUP_HPP_

#define _OPTIMIZER_SETUP_HPP_

#include <memory>     // for shared_ptr
#include <optional>   // for optional

#include "typeAliases.hpp"

namespace setup
{
    void setupOptimizer(pq::Engine &);

    /**
     * @class OptimizerSetup
     *
     * @brief Setup optimizer
     *
     */
    class OptimizerSetup
    {
       private:
        pq::OptEngine &_optEngine;

       public:
        explicit OptimizerSetup(pq::OptEngine &optEngine);

        void setup();
        void writeSetupInfo() const;

        void setupConvergence(pq::SharedOptimizer &);
        void setupMinMaxLR(pq::SharedLearningRate &);

        pq::SharedOptimizer    setupEmptyOptimizer();
        pq::SharedLearningRate setupLearningRateStrategy();
        pq::SharedEvaluator    setupEvaluator();
    };

}   // namespace setup

#endif   // _OPTIMIZER_SETUP_HPP_