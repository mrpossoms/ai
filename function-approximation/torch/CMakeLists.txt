cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(approx)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(FetchContent REQUIRED)
FetchContent_Declare(
  ascii-chart
  GIT_REPOSITORY https://github.com/Civitasv/asciichart.git
  GIT_TAG        master
)

FetchContent_MakeAvailable(ascii-chart)

add_executable(approx sin.cpp)
target_include_directories(approx PUBLIC ${CMAKE_BINARY_DIR}/_deps/ascii-chart-src/include)
target_link_libraries(approx "${TORCH_LIBRARIES}")
target_link_libraries(approx ncurses pthread m)
set_property(TARGET approx PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET approx
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)

