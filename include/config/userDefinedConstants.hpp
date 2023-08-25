// #ifndef _USER_DEFINED_CONSTANTS_HPP_

// #ifndef _USER_DEFINED_CONSTANTS_HPP_

// #define _USER_DEFINED_CONSTANTS_HPP_

// #include "defaults.hpp"

// #include <cstddef>

// namespace constants
// {
//     class UserDefinedConstants
//     {
//       private:
//         static inline size_t _shakeMaxIter  = defaults::_SHAKE_MAX_ITER_DEFAULT_;
//         static inline size_t _rattleMaxIter = defaults::_RATTLE_MAX_ITER_DEFAULT_;

//         static inline double _shakeTolerance  = defaults::_SHAKE_TOLERANCE_DEFAULT_;
//         static inline double _rattleTolerance = defaults::_RATTLE_TOLERANCE_DEFAULT_;

//       public:
//         [[nodiscard]] static size_t getShakeMaxIter() { return _shakeMaxIter; }
//         [[nodiscard]] static size_t getRattleMaxIter() { return _rattleMaxIter; }
//         [[nodiscard]] static double getShakeTolerance() { return _shakeTolerance; }
//         [[nodiscard]] static double getRattleTolerance() { return _rattleTolerance; }
//     };
// }   // namespace constants

// #endif   // _USER_DEFINED_CONSTANTS_HPP_