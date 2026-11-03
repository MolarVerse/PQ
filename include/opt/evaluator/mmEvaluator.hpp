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

#ifndef OPT__EVALUATOR__MM_EVALUATOR_HPP
#define OPT__EVALUATOR__MM_EVALUATOR_HPP

#include "evaluator.hpp"

namespace opt
{
    /**
     * @class MMEvaluator
     *
     * @brief Evaluator to evaluate forces/hessians using MM
     *
     */
    class MMEvaluator : public Evaluator
    {
       public:
        using Evaluator::Evaluator;

        [[nodiscard]] std::shared_ptr<Evaluator> clone() const override;
        void                                     evaluate() override;
    };

}   // namespace opt

#endif   // OPT__EVALUATOR__MM_EVALUATOR_HPP
