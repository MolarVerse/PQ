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
#include "vector3d.hpp"

namespace
{
    using linearAlgebra::tensor3D;
    using linearAlgebra::Vec3D;
    using molsys::OrthorhombicBox;
    using molsys::TriclinicBox;

    OrthorhombicBox makeOrthorhombicBox()
    {
        OrthorhombicBox box;
        box.setBoxDimensions({20.0, 25.0, 30.0});
        return box;
    }

    TriclinicBox makeTriclinicBox()
    {
        TriclinicBox box;
        box.setBoxDimensions({20.0, 25.0, 30.0});
        box.setBoxAngles({80.0, 90.0, 100.0});
        return box;
    }

    void BM_OrthorhombicShiftVector(benchmark::State& state)
    {
        auto  box = makeOrthorhombicBox();
        Vec3D vector{17.3, -14.1, 21.8};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(vector);
            auto result = box.calcShiftVector(vector);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicShiftVector(benchmark::State& state)
    {
        auto  box = makeTriclinicBox();
        Vec3D vector{17.3, -14.1, 21.8};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(vector);
            auto result = box.calcShiftVector(vector);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_OrthorhombicWrapPosition(benchmark::State& state)
    {
        auto  box = makeOrthorhombicBox();
        Vec3D position{27.3, -34.1, 41.8};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(position);
            auto result = box.wrapPositionIntoBox(position);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicWrapPosition(benchmark::State& state)
    {
        auto  box = makeTriclinicBox();
        Vec3D position{27.3, -34.1, 41.8};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(position);
            auto result = box.wrapPositionIntoBox(position);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicCoordinateRoundTrip(benchmark::State& state)
    {
        auto  box = makeTriclinicBox();
        Vec3D vector{3.2, -1.7, 4.6};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(vector);
            auto result = box.toSimSpace(box.toOrthoSpace(vector));
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicToOrthoSpace(benchmark::State& state)
    {
        auto  box = makeTriclinicBox();
        Vec3D vector{3.2, -1.7, 4.6};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(vector);
            auto result = box.toOrthoSpace(vector);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicToSimSpace(benchmark::State& state)
    {
        auto  box = makeTriclinicBox();
        Vec3D vector{3.2, -1.7, 4.6};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(vector);
            auto result = box.toSimSpace(vector);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_TriclinicTensorRoundTrip(benchmark::State& state)
    {
        auto     box = makeTriclinicBox();
        tensor3D tensor{
            Vec3D{2.0, 0.1, 0.2},
            Vec3D{0.3, 3.0, 0.1},
            Vec3D{0.2, 0.1, 4.0}
        };

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(tensor);
            auto result = box.toSimSpace(box.toOrthoSpace(tensor));
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
