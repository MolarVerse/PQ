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

#include "staticMatrix.hpp"
#include "vector3d.hpp"

namespace
{
    using linearAlgebra::StaticMatrix3x3;
    using linearAlgebra::Vec3D;

    void BM_VectorArithmetic(benchmark::State& state)
    {
        Vec3D lhs{1.1, 2.2, 3.3};
        Vec3D rhs{0.7, -1.3, 2.1};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(lhs);
            benchmark::DoNotOptimize(rhs);
            auto result = (lhs + rhs) * 0.5 - rhs;
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_DotProduct(benchmark::State& state)
    {
        Vec3D lhs{1.1, 2.2, 3.3};
        Vec3D rhs{0.7, -1.3, 2.1};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(lhs);
            benchmark::DoNotOptimize(rhs);
            auto result = dot(lhs, rhs);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_CrossProduct(benchmark::State& state)
    {
        Vec3D lhs{1.1, 2.2, 3.3};
        Vec3D rhs{0.7, -1.3, 2.1};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(lhs);
            benchmark::DoNotOptimize(rhs);
            auto result = cross(lhs, rhs);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_VectorNorm(benchmark::State& state)
    {
        Vec3D vector{1.1, 2.2, 3.3};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(vector);
            auto result = norm(vector);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_MatrixVectorProduct(benchmark::State& state)
    {
        StaticMatrix3x3<double> matrix{
            Vec3D{2.0, 0.1, 0.2},
            Vec3D{0.3, 3.0, 0.1},
            Vec3D{0.2, 0.1, 4.0}
        };
        Vec3D vector{1.1, 2.2, 3.3};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(matrix);
            benchmark::DoNotOptimize(vector);
            auto result = matrix * vector;
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_MatrixInverse(benchmark::State& state)
    {
        StaticMatrix3x3<double> matrix{
            Vec3D{2.0, 0.1, 0.2},
            Vec3D{0.3, 3.0, 0.1},
            Vec3D{0.2, 0.1, 4.0}
        };

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(matrix);
            auto result = inverse(matrix);
            benchmark::DoNotOptimize(result);
        }
    }

    BENCHMARK(BM_VectorArithmetic);
    BENCHMARK(BM_DotProduct);
    BENCHMARK(BM_CrossProduct);
    BENCHMARK(BM_VectorNorm);
    BENCHMARK(BM_MatrixVectorProduct);
    BENCHMARK(BM_MatrixInverse);
}   // namespace
