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

#ifndef _ASE_DFTB_RUNNER_HPP_

#define _ASE_DFTB_RUNNER_HPP_

#include <unordered_map>   // for unordered_map

#include "aseQMRunner.hpp"   // for InternalQMRunner

namespace QM
{
    /**
     * @brief AseDftbRunner inherits from ASEQMRunner
     *
     */
    class __attribute__((visibility("default"))) AseDftbRunner
        : public ASEQMRunner
    {
       public:
        ~AseDftbRunner() override = default;

        explicit AseDftbRunner(
            const std::string &slakosType,
            const std::string &slakosPath,
            const bool         thirdOrder
        );


        /***************************
         * standard getter methods *
         ***************************/
        const std::unordered_map<std::string, double> getHubbDerivDict() const;


        /***************************
         * standard setter methods *
         ***************************/
        void setHubbDerivDict(const std::unordered_map<std::string, double> slakosDict);


       private:
        std::unordered_map<std::string, double> _hubbardDerivDict;
    };
}   // namespace QM

#endif   // _ASE_DFTB_RUNNER_HPP_