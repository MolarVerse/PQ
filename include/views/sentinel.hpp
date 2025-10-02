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

#ifndef __SENTINEL_HPP__
#define __SENTINEL_HPP__

#include <iterator>      // for std::iterator_traits
#include <type_traits>   // for std::conditional_t

namespace pqviews
{
    namespace detail
    {
        template <typename T>
        struct has_difference_type
        {
           private:
            template <typename U>
            static auto test(
                int
            ) -> decltype(typename std::iterator_traits<U>::difference_type{}, std::true_type{});

            template <typename>
            static std::false_type test(...);

           public:
            using type                  = decltype(test<T>(0));
            static constexpr bool value = type::value;
        };

        template <typename T>
        constexpr bool has_difference_type_v = has_difference_type<T>::value;

        template <typename T, bool HasDiffType = has_difference_type_v<T>>
        struct difference_type_helper
        {
            using type = std::ptrdiff_t;
        };

        template <typename T>
        struct difference_type_helper<T, true>
        {
            using type = typename std::iterator_traits<T>::difference_type;
        };

        template <typename T>
        using difference_type_or_ptrdiff =
            typename difference_type_helper<T>::type;
    }   // namespace detail

    /**
     * @brief Sentinel is a generic sentinel for custom iterators
     *
     * @tparam Iter The iterator type to be used as end marker
     */
    template <typename Iter>
    class Sentinel
    {
       private:
        Iter _end;

       public:
        // Use iterator_traits if available, otherwise fall back to
        // std::ptrdiff_t
        using difference_type = detail::difference_type_or_ptrdiff<Iter>;

        // make it default constructible, copyable, and movable
        Sentinel()                           = default;
        Sentinel(const Sentinel&)            = default;
        Sentinel(Sentinel&&)                 = default;
        Sentinel& operator=(const Sentinel&) = default;
        Sentinel& operator=(Sentinel&&)      = default;

        // Required by std::sentinel_for
        explicit Sentinel(Iter end) : _end(end) {}

        /**
         * @brief checks if the iterator is equal to the sentinel
         *
         * @param it the Iterator to compare with
         * @return true if they are equal, false otherwise
         */
        template <typename Iterator>
        friend bool operator==(const Iterator& it, const Sentinel& s)
        {
            return it.current() == s._end;
        }

        /**
         * @brief checks if the sentinel is equal to the iterator
         *
         * @param it the Iterator to compare with
         * @return true if they are equal, false otherwise
         */
        template <typename Iterator>
        friend bool operator!=(const Iterator& it, const Sentinel& s)
        {
            return !(it == s);
        }

        /**
         * @brief calculates the distance between the sentinel and the iterator
         *
         * @param s the Sentinel to compare with
         * @param it the Iterator to compare with
         * @return difference_type the distance between the two
         */
        template <typename Iterator>
        friend difference_type operator-(const Sentinel& s, const Iterator& it)
        {
            Iterator        temp  = it;
            difference_type count = 0;

            while (temp != s)
            {
                ++temp;
                ++count;
            }

            return count;
        }

        /**
         * @brief calculates the distance between the iterator and the sentinel
         *
         * @param it the Iterator to compare with
         * @param s the Sentinel to compare with
         * @return difference_type the distance between the two
         */
        template <typename Iterator>
        friend difference_type operator-(const Iterator& it, const Sentinel& s)
        {
            return -(s - it);
        }
    };

}   // namespace pqviews

#endif   // __SENTINEL_HPP__
