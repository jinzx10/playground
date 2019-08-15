# Install script for directory: /data/home/jinzx10/playground/cmake/hw3

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/home/jinzx10/playground/cmake/hw3/bin/main")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/data/home/jinzx10/playground/cmake/hw3/bin" TYPE EXECUTABLE FILES "/data/home/jinzx10/playground/cmake/hw3/build/bin/main")
  if(EXISTS "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main"
         OLD_RPATH "/data/home/jinzx10/playground/cmake/hw3/lib:/data/home/jinzx10/playground/cmake/hw3/build/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main2" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main2")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main2"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/home/jinzx10/playground/cmake/hw3/bin/main2")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/data/home/jinzx10/playground/cmake/hw3/bin" TYPE EXECUTABLE FILES "/data/home/jinzx10/playground/cmake/hw3/build/bin/main2")
  if(EXISTS "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main2" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main2")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main2"
         OLD_RPATH "/data/home/jinzx10/playground/cmake/hw3/lib:/data/home/jinzx10/playground/cmake/hw3/build/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/data/home/jinzx10/playground/cmake/hw3/bin/main2")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/data/home/jinzx10/playground/cmake/hw3/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
