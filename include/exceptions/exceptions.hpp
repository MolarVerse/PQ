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

#ifndef _EXCEPTIONS_HPP_

#define _EXCEPTIONS_HPP_

#include <cstddef>
#include <exception>
#include <string>
#include <string_view>

#include "color.hpp"

namespace customException
{

    /**
     * @enum ExceptionType
     *
     */
    enum class ExceptionType : size_t
    {
        INPUTFILEEXCEPTION,
        RSTFILEEXCEPTION,
        USERINPUTEXCEPTION,
        MOLDESCRIPTOREXCEPTION,
        USERINPUTEXCEPTIONWARNING,
        GUFFDATEXCEPTION,
        TOPOLOGYEXCEPTION,
        PARAMETERFILEEXCEPTION,
        MANOSTATEXCEPTION,
        INTRANONBONDEDEXCEPTION,
        SHAKEEXCEPTION,
        CELLLISTEXCEPTION,
        RINGPOLYMERRESTARTFILEEXCEPTION,
        QMRUNNEREXCEPTION,
        MPIEXCEPTION,
        QMRUNTIMEEXCEEDED,
        MSHAKEFILEEXCEPTION,
        MSHAKEEXCEPTION
    };

    /**
     * @class CustomException
     *
     * @brief Custom exception base class
     *
     */
    class CustomException : public std::exception
    {
       protected:
        std::string _message;

       public:
        explicit CustomException(const std::string_view message)
            : _message(message){};
        void colorfulOutput(const Color::Code color, const std::string_view)
            const;
    };

    /**
     * @class InputFileException inherits from CustomException
     *
     * @brief Exception for input file errors
     *
     */
    class InputFileException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class RstFileException inherits from CustomException
     *
     * @brief Exception for restart file errors
     *
     */
    class RstFileException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class UserInputException inherits from CustomException
     *
     * @brief Exception for user input errors (CLI)
     *
     */
    class UserInputException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class MolDescriptorException inherits from CustomException
     *
     * @brief Exception for MolDescriptor errors
     *
     */
    class MolDescriptorException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class UserInputExceptionWarning inherits from CustomException
     *
     * @brief Exception for user input warnings
     *
     */
    class UserInputExceptionWarning : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class GuffDatException inherits from CustomException
     *
     * @brief Exception for guff.dat errors
     *
     */
    class GuffDatException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class TopologyException inherits from CustomException
     *
     * @brief Exception for topology file errors
     */
    class TopologyException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class ParameterFileException inherits from CustomException
     *
     * @brief Exception for parameter file errors
     */
    class ParameterFileException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class ManostatException inherits from CustomException
     *
     * @brief Exception for manostat errors
     */
    class ManostatException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class IntraNonBondedException inherits from CustomException
     *
     * @brief Exception for intra non bonded errors
     */
    class IntraNonBondedException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class ShakeException inherits from CustomException
     *
     * @brief Exception for SHAKE errors
     */
    class ShakeException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class CellListException inherits from CustomException
     *
     * @brief Exception for CellList errors
     */
    class CellListException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class RingPolymerRestartFileException inherits from CustomException
     *
     * @brief Exception for ring polymer restart file errors
     */
    class RingPolymerRestartFileException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class QMRunnerException inherits from CustomException
     *
     * @brief Exception for QMRunner errors
     */
    class QMRunnerException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class MPIException inherits from CustomException
     *
     * @brief Exception for MPI errors
     *
     */
    class MPIException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class QMRunTimeExceeded inherits from CustomException
     *
     * @brief Exception for QM runtime exceeded
     *
     */
    class QMRunTimeExceeded : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class MShakeFileException inherits from CustomException
     *
     * @brief Exception for mShake errors
     */
    class MShakeFileException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

    /**
     * @class MShakeException inherits from CustomException
     *
     * @brief Exception for MShake errors
     */
    class MShakeException : public CustomException
    {
       public:
        using CustomException::CustomException;

        const char *what() const throw() override;
    };

}   // namespace customException

#endif   // _EXCEPTIONS_HPP_