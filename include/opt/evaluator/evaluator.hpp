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

#ifndef _EVALUATOR_HPP_

#define _EVALUATOR_HPP_

namespace opt
{
    /**
     * @class Evaluator
     *
     * @brief Base class for all evaluators (e.g. MM, QM, ...)
     *        Evaluators are used to evaluate forces/hessians
     *
     */
    class Evaluator
    {
       public:
        Evaluator()          = default;
        virtual ~Evaluator() = default;
    };

}   // namespace opt

#endif   // _EVALUATOR_HPP_