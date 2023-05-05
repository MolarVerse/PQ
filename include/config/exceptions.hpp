#ifndef _EXCEPTIONS_H_

#define _EXCEPTIONS_H_

#include <exception>
#include <string>
#include <iostream>

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
    explicit CustomException(std::string_view message) : _message(message){};
    void colorfulOutput(Color::Code color, std::string_view) const;
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
    explicit InputFileException(const std::string &message) : CustomException(message){};

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
    explicit RstFileException(const std::string &message) : CustomException(message){};

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
    explicit UserInputException(const std::string &message) : CustomException(message){};

    const char *what() const throw() override;
};

#endif