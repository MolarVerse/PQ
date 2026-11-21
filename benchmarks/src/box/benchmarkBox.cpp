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

#include <benchmark/benchmark.h>

#include "orthorhombicBox.hpp"
#include "staticMatrix.hpp"
#include "triclinicBox.hpp"
#include "units.hpp"
#include "vector3d.hpp"

namespace
{
    molsys::OrthorhombicBox makeOrthorhombicBox()
    {
        molsys::OrthorhombicBox box(
            {20.0 * angstrom, 25.0 * angstrom, 30.0 * angstrom}
        );
        return box;
    }

    molsys::TriclinicBox makeTriclinicBox()
    {
        molsys::TriclinicBox box{
            {20.0 * angstrom, 25.0 * angstrom, 30.0 * angstrom},
            linalg::Vec3D{80.0, 90.0, 100.0} * deg,
            false
        };
        return box;
    }

    void BM_OrthorhombicShiftVector(benchmark::State& state)
    {
        auto  box    = makeOrthorhombicBox();
        Pos3D vector = linalg::Vec3D{17.3, -14.1, 21.8} * angstrom;

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(vector);
            auto result = box.calcShiftVector(vector);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicShiftVector(benchmark::State& state)
    {
        auto  box    = makeTriclinicBox();
        Pos3D vector = linalg::Vec3D{17.3, -14.1, 21.8} * angstrom;

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(vector);
            auto result = box.calcShiftVector(vector);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_OrthorhombicWrapPosition(benchmark::State& state)
    {
        auto  box      = makeOrthorhombicBox();
        Pos3D position = linalg::Vec3D{27.3, -34.1, 41.8} * angstrom;

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(position);
            auto result = box.wrapPositionIntoBox(position);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicWrapPosition(benchmark::State& state)
    {
        auto  box      = makeTriclinicBox();
        Pos3D position = linalg::Vec3D{27.3, -34.1, 41.8} * angstrom;

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(position);
            auto result = box.wrapPositionIntoBox(position);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicCoordinateRoundTrip(benchmark::State& state)
    {
        auto  box    = makeTriclinicBox();
        Pos3D vector = linalg::Vec3D{3.2, -1.7, 4.6} * angstrom;

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(vector);
            auto result =
                box.getBoxMatrix() * (box.getInverseBoxMatrix() * vector);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicToOrthoSpace(benchmark::State& state)
    {
        auto  box    = makeTriclinicBox();
        Pos3D vector = linalg::Vec3D{3.2, -1.7, 4.6} * angstrom;

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(vector);
            auto result = box.getInverseBoxMatrix() * vector;
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicToSimSpace(benchmark::State& state)
    {
        auto box    = makeTriclinicBox();
        auto vector = linalg::Vec3D{3.2, -1.7, 4.6};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(vector);
            auto result = box.getBoxMatrix() * vector;
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicTensorRoundTrip(benchmark::State& state)
    {
        auto             box = makeTriclinicBox();
        linalg::tensor3D tensor{
            linalg::Vec3D{2.0, 0.1, 0.2},
            linalg::Vec3D{0.3, 3.0, 0.1},
            linalg::Vec3D{0.2, 0.1, 4.0}
        };

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(tensor);
            auto result =
                box.getBoxMatrix() * (box.getInverseBoxMatrix() * tensor);
            benchmark::DoNotOptimize(result);
        }
    }

    BENCHMARK(BM_OrthorhombicShiftVector);
    BENCHMARK(BM_TriclinicShiftVector);
    BENCHMARK(BM_OrthorhombicWrapPosition);
    BENCHMARK(BM_TriclinicWrapPosition);
    BENCHMARK(BM_TriclinicCoordinateRoundTrip);
    BENCHMARK(BM_TriclinicToOrthoSpace);
    BENCHMARK(BM_TriclinicToSimSpace);
    BENCHMARK(BM_TriclinicTensorRoundTrip);

}   // namespace
