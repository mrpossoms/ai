cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(puck)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(FetchContent REQUIRED)
FetchContent_Declare(
  spare-time
  GIT_REPOSITORY https://github.com/mrpossoms/spare-time.git
  GIT_TAG        master
)

FetchContent_MakeAvailable(spare-time)

add_library(puck-core policy.cpp env.cpp)
target_link_libraries(puck-core "${TORCH_LIBRARIES}")
target_link_libraries(puck-core ncurses pthread m)
target_include_directories(puck-core PUBLIC ${CMAKE_BINARY_DIR}/_deps/spare-time-src)

add_executable(puck driver.cpp)
target_link_libraries(puck puck-core)
set_property(TARGET puck PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET puck
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)

# Setup testing
enable_testing()
add_executable(tests tests.cpp)
target_link_libraries(tests puck-core)
set_property(TARGET tests PROPERTY CXX_STANDARD 17)
