#ifndef _EXCEPTIONS_H_

#define _EXCEPTIONS_H_

#include <exception>
#include <string>
#include <iostream>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "color.hpp"

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
    explicit CustomException(const std::string_view message) : _message(message){};
    void colorfulOutput(const Color::Code color, const std::string_view) const;
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

#endif