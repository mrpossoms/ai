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

add_executable(puck driver.cpp net.cpp rl.cpp env.cpp)
target_include_directories(puck PUBLIC ${CMAKE_BINARY_DIR}/_deps/spare-time-src)
target_link_libraries(puck "${TORCH_LIBRARIES}")
target_link_libraries(puck ncurses pthread m)
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
add_executable(tests tests.cpp net.cpp rl.cpp env.cpp)
target_include_directories(tests PUBLIC ${CMAKE_BINARY_DIR}/_deps/spare-time-src)
target_link_libraries(tests ncurses pthread m)
target_link_libraries(tests "${TORCH_LIBRARIES}")