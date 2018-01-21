# This file is part of GPUraku examples.
# 
# GPUraku is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# GPUraku is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with GPUraku.  If not, see <http://www.gnu.org/licenses/>.

## Specify the name of the example project
set( EXECUTABLE_NAME decode )

## Compiler configuration
# Set the compiling flags.
set( CMAKE_C_FLAGS "-Wall -Wextra" )
set( CMAKE_C_FLAGS_DEBUG "-g -O0" )
set( CMAKE_C_FLAGS_RELEASE "-O3 -mmmx -msse -msse2 -msse3" )
# Specify the name of dependent libraries
set ( LIBGPURAKU gpuraku stdc++ )
# ## Dependencies - OpenMP library required.
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)  
  message("-- Setting OpenMP flags for compiler...")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()  

## Library usage
# Add headers files of GPUraku.
include_directories(../../include)
# Add source code of the example.
aux_source_directory(. DIR_SRCS)

# Set the project output binaries.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../bin/)
add_executable(${EXECUTABLE_NAME} ${DIR_SRCS})
# Linked with libraries.
target_link_libraries(${EXECUTABLE_NAME} ${LIBGPURAKU})