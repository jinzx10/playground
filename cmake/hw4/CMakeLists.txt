cmake_minimum_required(VERSION 2.8)

set(ROOTDIR ~/playground/cmake/hw4 )
set(SRCDIR ${ROOTDIR}/src)
set(BUILDDIR ${ROOTDIR}/build)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BUILDDIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BUILDDIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILDDIR}/bin)

include_directories(${ROOTDIR}/include)
link_directories(${ROOTDIR}/lib)

add_executable(main ${SRCDIR}/main.cpp ${SRCDIR}/tq.cpp)
target_link_libraries(main dfact fact fib)

add_executable(main2 ${SRCDIR}/main.cpp ${SRCDIR}/tq.cpp)
target_link_libraries(main2 dfact fact fib)

add_library(dfact STATIC ${SRCDIR}/dfact.cpp)
add_library(fact STATIC ${SRCDIR}/fact.cpp)
add_library(fib SHARED ${SRCDIR}/fib.cpp)

install(TARGETS main main2 DESTINATION ${ROOTDIR}/bin)
