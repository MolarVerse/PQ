# - Returns a version string from Git tags
#
# This function inspects the annotated git tags for the project and returns a string
# into a CMake variable
#
# get_git_version(<var>)
#
# - Example
#
# include(GetGitVersion)
# get_git_version(GIT_VERSION)
#
# Requires CMake 2.8.11+
find_package(Git)

if(GIT_EXECUTABLE)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --match "v[0-9]*.[0-9]*.[0-9]*" --abbrev=8 --tags
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE status
        OUTPUT_VARIABLE GIT_VERSION
        ERROR_QUIET)

    if(${status})
        set(GIT_VERSION "v0.0.0")
    else()
        string(STRIP ${GIT_VERSION} GIT_VERSION)
        # v1.1.0-67-g67380c03  ->  v1.1.0+67380c03
        string(REGEX REPLACE "-[0-9]+-g" "+" GIT_VERSION ${GIT_VERSION})
    endif()

    # A release is prepared on the branch release/vX.Y.Z, which exists before
    # its tag does (the tag is created when the release is published), so
    # git describe would still give the previous release. Take the version from
    # the branch name instead: v1.2.0-<hash> as for any build after a tag. Pull
    # request checkouts of CI are detached, then the name of the branch of the
    # pull request is in GITHUB_HEAD_REF.
    set(PQ_GIT_BRANCH "")
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE branch_status
        OUTPUT_VARIABLE PQ_GIT_BRANCH
        ERROR_QUIET)
    string(STRIP "${PQ_GIT_BRANCH}" PQ_GIT_BRANCH)

    if(branch_status OR PQ_GIT_BRANCH STREQUAL "HEAD")
        set(PQ_GIT_BRANCH "")
        foreach(branch_variable GITHUB_HEAD_REF GITHUB_REF_NAME)
            if(PQ_GIT_BRANCH STREQUAL "" AND NOT "$ENV{${branch_variable}}" STREQUAL "")
                set(PQ_GIT_BRANCH "$ENV{${branch_variable}}")
            endif()
        endforeach()
    endif()

    if(PQ_GIT_BRANCH MATCHES "^release/(v[0-9]+\\.[0-9]+\\.[0-9]+)$")
        set(release_version "${CMAKE_MATCH_1}")
        execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short=8 HEAD
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE hash_status
            OUTPUT_VARIABLE release_hash
            ERROR_QUIET)
        string(STRIP "${release_hash}" release_hash)

        if(hash_status OR release_hash STREQUAL "")
            set(GIT_VERSION "${release_version}")
        else()
            set(GIT_VERSION "${release_version}-${release_hash}")
        endif()
    endif()

    # Work out if the repository is dirty
    execute_process(COMMAND ${GIT_EXECUTABLE} update-index -q --refresh
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_QUIET
        ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} diff-index --name-only HEAD --
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_DIFF_INDEX
        ERROR_QUIET)
    string(COMPARE NOTEQUAL "${GIT_DIFF_INDEX}" "" GIT_DIRTY)

    if(${GIT_DIRTY})
        set(GIT_VERSION "${GIT_VERSION}")
    endif()
else()
    set(GIT_VERSION "v0.0.0")
endif()

if(GIT_VERSION STREQUAL "v0.0.0")
    message(WARNING
        "No git tag of the form vX.Y.Z is reachable (a source archive, a "
        "clone without tags or no git at all): the version of this build is "
        "unknown (v0.0.0) and checks that depend on it, such as the removal "
        "of deprecated code, cannot fire. Fetch the tags (git fetch --tags) "
        "or build from a clone of the repository.")
endif()

message("-- git Version: ${GIT_VERSION}")
