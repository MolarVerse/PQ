"""
*****************************************************************************
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
*****************************************************************************
"""

'''Common things needed by command line scripts.'''


class ToolError(Exception):
    '''Exception thrown by command line tool.'''


def print_header():

    print("*************************************************************************")
    print("*                                                                       *")
    print("*                            _                                    ___   *")
    print("*          _                ( )                                 /'___)  *")
    print("*   _ _   (_)  ___ ___     _| | ______   _ _   ___ ___     ___ | (__    *")
    print("*  ( '_`\ | |/' _ ` _ `\ /'_` |(______)/'_` )/' _ ` _ `\ /'___)| ,__)   *")
    print("*  | (_) )| || ( ) ( ) |( (_| |       ( (_) || ( ) ( ) |( (___ | |      *")
    print("*  | ,__/'(_)(_) (_) (_)`\__,_)       `\__, |(_) (_) (_)`\____)(_)      *")
    print("*  | |                                    | |                           *")
    print("*  (_)                                    (_)                           *")
    print("*                                                                       *")
    print("*                                                                       *")
    print("*************************************************************************")
    print("")
