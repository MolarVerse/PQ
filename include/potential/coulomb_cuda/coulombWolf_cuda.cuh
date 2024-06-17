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

#ifndef _CUDA_COULOMB_WOLF_HPP_
#define _CUDA_COULOMB_WOLF_HPP_

#include <cuda_runtime.h>
#include <utility>   // for pair

namespace potential
{
    /**
     * @brief Structure for the CoulombWolf potential on the device
     */
    struct CudaCoulombWolf_t
    {
        double coulombRadiusCutOff;
        double kappa;
        double wolfParameter1;
        double wolfParameter2;
        double wolfParameter3;
        double prefactor;
    };   // struct CudaCoulombWolf_t

    /**
     * @class CoulombWolf
     *
     * @brief
     * CoulombWolf inherits CoulombPotential
     * CoulombWolf is a class for the Coulomb potential with Wolf summation as
     * long range correction
     *
     */
    class CudaCoulombWolf
    {
    private:
        double _coulombRadiusCutOff;
        double _kappa;
        double _wolfParameter1;
        double _wolfParameter2;
        double _wolfParameter3;
        double _prefactor;

    public:
        CudaCoulombWolf(
            const double coulombRadiusCutOff,
            const double kappa,
            const double wolfParameter1,
            const double wolfParameter2,
            const double wolfParameter3,
            const double prefactor
        );

        CudaCoulombWolf() = default;
        ~CudaCoulombWolf() = default;

        CudaCoulombWolf_t* getCudaCoulombWolf() const;

        double getCoulombRadiusCutOff() const
        {
            return _coulombRadiusCutOff;
        }
    };

    __device__ void calculateWolfKernel(
        CudaCoulombWolf_t* coulombWolf,
        const double distance,
        const double charge_i,
        const double charge_j,
        double& force,
        double* coulombEnergy
    );

}   // namespace potential

#endif   // _CUDA_COULOMB_WOLF_HPP_
