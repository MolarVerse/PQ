#ifndef _COLOR_HPP_

#define _COLOR_HPP_

#include <ostream>
namespace Color
{
    /**
     * @enum Code
     *
     * @brief ANSI escape codes for colors
     *
     */
    enum Code
    {
        FG_RED     = 31,
        FG_GREEN   = 32,
        FG_BLUE    = 34,
        FG_ORANGE  = 33,
        FG_DEFAULT = 39,
        BG_RED     = 41,
        BG_GREEN   = 42,
        BG_BLUE    = 44,
        BG_DEFAULT = 49
    };

    /**
     * @class Modifier
     *
     * @brief Modifier class for ANSI escape codes
     *
     */
    class Modifier
    {
        Code code;

      public:
        explicit Modifier(const Code pCode) : code(pCode) {}
        friend std::ostream &operator<<(std::ostream &os, const Modifier &mod) { return os << "\033[" << mod.code << "m"; }
    };
}   // namespace Color

#endif   // _COLOR_HPP_