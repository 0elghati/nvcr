include(FetchContent)

find_package(spdlog 1.11 QUIET)
find_package(fmt 9 QUIET)

if(NVCR_FETCH_DEPENDENCIES)
    if(NOT spdlog_FOUND)
        FetchContent_Declare(
            spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG v1.14.1
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(spdlog)
    endif()

    if(NVCR_BUILD_EXAMPLES)
        find_package(CLI11 2.3 QUIET)
        if(NOT CLI11_FOUND)
            FetchContent_Declare(
                cli11
                GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
                GIT_TAG v2.4.2
                GIT_SHALLOW TRUE)
            FetchContent_MakeAvailable(cli11)
        endif()
    endif()

    if(NVCR_BUILD_TESTS)
        find_package(GTest 1.12 QUIET)
        if(NOT GTest_FOUND)
            FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG v1.14.0
                GIT_SHALLOW TRUE)
            set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
            FetchContent_MakeAvailable(googletest)
        endif()
    endif()
endif()

if(NVCR_BUILD_TESTS AND NOT TARGET GTest::gtest_main)
    find_package(GTest 1.12 QUIET)
endif()

if(NVCR_ENABLE_OPENCV)
    find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)
endif()
