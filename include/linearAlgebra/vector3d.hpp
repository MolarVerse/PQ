#ifndef _VECTOR3d_HPP_

#define _VECTOR3d_HPP_

#include "concepts.hpp"
#include "vector3dClass.hpp"

namespace linearAlgebra
{
    /************************
     * comparison operators *
     ************************/

    template <class U>
    requires std::equality_comparable<U>
    bool operator==(const Vector3D<U> &lhs, const Vector3D<U> &rhs);

    /*********************
     * binary + operator *
     *********************/

    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> == pq::Vector3DDepth_v<V>)
    auto operator+(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] + rhs[0])>;

    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> - 1 == pq::Vector3DDepth_v<V>)
    auto operator+(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] + rhs)>;

    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> == pq::Vector3DDepth_v<V> - 1)
    auto operator+(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs + rhs[0])>;

    template <pq::Vector3DConcept U, pq::Arithmetic V>
    auto operator+(const U &vec, const V &scalar)
        -> Vector3D<decltype(vec[0] + scalar)>;

    template <pq::Arithmetic U, pq::Vector3DConcept V>
    auto operator+(const U &scalar, const V &vec)
        -> Vector3D<decltype(vec[0] + scalar)>;

    /*********************
     * binary - operator *
     *********************/

    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> == pq::Vector3DDepth_v<V>)
    auto operator-(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] - rhs[0])>;

    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> - 1 == pq::Vector3DDepth_v<V>)
    auto operator-(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs[0] - rhs)>;

    template <pq::Vector3DConcept U, pq::Vector3DConcept V>
    requires(pq::Vector3DDepth_v<U> == pq::Vector3DDepth_v<V> - 1)
    auto operator-(const U &lhs, const V &rhs)
        -> Vector3D<decltype(lhs - rhs[0])>;

    template <pq::Vector3DConcept U, pq::Arithmetic V>
    auto operator-(const U &vec, const V &scalar)
        -> Vector3D<decltype(vec[0] - scalar)>;

    template <pq::Arithmetic U, pq::Vector3DConcept V>
    auto operator-(const U &scalar, const V &vec)
        -> Vector3D<decltype(vec[0] - scalar)>;

}   // namespace linearAlgebra

#include "vector3d.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _VECTOR3d_HPP_