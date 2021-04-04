cmake_minimum_required(VERSION 2.8.12)
project(rle_bench_vec)

find_package(benchmark REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g")

add_compile_definitions(SIZE=${SIZE})
add_compile_definitions(REPEAT=${REPEAT})
add_compile_definitions(LOWER_RLE=${LOWER_RLE})
add_compile_definitions(UPPER_RLE=${UPPER_RLE})
add_compile_definitions(UPPER_VAL=${UPPER_VAL})
add_compile_definitions(RLE_BITS=${RLE_BITS})

add_executable(${PROJECT_NAME} bench.cpp vec_dense.c vec_rle.c)

# To let the app be a standalone project
if(NOT TACO_INCLUDE_DIR)
  if(NOT DEFINED ENV{TACO_INCLUDE_DIR} OR NOT DEFINED ENV{TACO_LIBRARY_DIR})
    message(FATAL_ERROR "Set the environment variables TACO_INCLUDE_DIR and TACO_LIBRARY_DIR")
  endif()
  set(TACO_INCLUDE_DIR $ENV{TACO_INCLUDE_DIR})
  set(TACO_LIBRARY_DIR $ENV{TACO_LIBRARY_DIR})
  find_library(taco taco ${TACO_LIBRARY_DIR})
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${taco} benchmark::benchmark)
else()
  set_target_properties("${PROJECT_NAME}" PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC taco benchmark::benchmark)
endif()

# Include taco headers
include_directories(${TACO_INCLUDE_DIR})