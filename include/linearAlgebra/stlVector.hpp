#ifndef _STL_VECTOR_HPP_

#define _STL_VECTOR_HPP_

#include <algorithm>   // for max_element
#include <numeric>     // for accumulate
#include <vector>      // for vector

namespace std
{
    /**
     * @brief Calculates the sum of all elements in a vector
     *
     * @param vector
     *
     * @return sum of all elements in vector
     */
    template <typename T>
    T sum(const std::vector<T> &vector)
    {
        return std::accumulate(vector.begin(), vector.end(), T());
    }

    /**
     * @brief Calculates the mean of all elements in a vector
     *
     * @param vector
     *
     * @return mean of all elements in vector
     */
    template <typename T>
    double mean(const std::vector<T> &vector)
    {
        return sum(vector) / double(vector.size());
    }

    /**
     * @brief Calculates the maximum of all elements in a vector
     *
     * @param vector
     *
     * @return maximum of all elements in vector
     */
    template <typename T>
    T max(const std::vector<T> &vector)
    {
        return *std::ranges::max_element(vector);
    }

}   // namespace std

#endif   // _STL_VECTOR_HPP_