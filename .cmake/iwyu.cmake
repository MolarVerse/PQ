if(NOT CMAKE_IWYU)
    set(CMAKE_IWYU OFF)
endif()

if(CMAKE_IWYU)
    find_program(iwyu_path NAMES include-what-you-use iwyu REQUIRED)

    set(iwyu_path_and_options
        ${iwyu_path}

        # -Xiwyu
        # --verbose=6
        # --mapping_file=${PROJECT_SOURCE_DIR}/.iwyu.imp
    )

    function(recursiveTargets DIR)
        get_property(TGTS DIRECTORY "${DIR}" PROPERTY BUILDSYSTEM_TARGETS)

        foreach(TGT IN LISTS TGTS)
            set_property(TARGET ${TGT}
                PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path_and_options})
        endforeach()

        get_property(SUBDIRS DIRECTORY "${DIR}" PROPERTY SUBDIRECTORIES)

        foreach(SUBDIR IN LISTS SUBDIRS)
            recursiveTargets("${SUBDIR}")
        endforeach()
    endfunction()

    recursiveTargets(src)
    recursiveTargets(tests)
    recursiveTargets(apps)
    recursiveTargets(tools)
endif()