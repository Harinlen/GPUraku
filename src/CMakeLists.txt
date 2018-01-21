# This file is part of GPUraku.
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

## Specify the name of the library
set( LIBRARY_NAME gpuraku )

# Configure the directories parameters.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin/)
set(GPURAKU_INCLUDE_DIRS ../include ./include)

## Dependencies - stdc++
# Set the c++ library.
set(CPP_LIBRARY stdc++)
# ## Dependencies - CUDA library required.
find_package( CUDA REQUIRED )
# ## Dependencies - OpenMP library required.
FIND_PACKAGE( OpenMP REQUIRED)  

# Add headers files.
include_directories( ${CUDA_INCLUDE_DIRS}
                     ${CMAKE_CURRENT_SOURCE_DIR}
                     ${GPURAKU_INCLUDE_DIRS}
                   )
## Add source code files.
## Split the source into a GPU part (compiled by nvcc)
file( GLOB GPU_SOURCES ./formats/flac/cuda/*.cu )
## and a CPU part (compiled by e.g. gcc).
aux_source_directory(./                  CPU_SOURCES)
aux_source_directory(./formats/flac      CPU_SOURCES)
aux_source_directory(./formats/flac/cuda CPU_SOURCES)
aux_source_directory(./formats/wav       CPU_SOURCES)
## (This may not always be necessary.)

## C++ compiler options
set( CMAKE_C_FLAGS "-Wall -Wextra" )
set( CMAKE_C_FLAGS_DEBUG "-g -O0" )
set( CMAKE_C_FLAGS_RELEASE "-O2 -mmmx -msse -msse2 -msse3 -ftree-vectorize -funroll-loops -ftree-loop-linear -floop-parallelize-all" )
if(OPENMP_FOUND)  
  message("-- Setting OpenMP flags for C/C++ compiler...")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()  


## NVCC (CUDA compiler) options
set( CUDA_NVCC_FLAGS_DEBUG -g -G -O0 )
set( CUDA_NVCC_FLAGS_RELEASE -O3 -lineinfo )


## 1. Compile...
cuda_add_library( ${LIBRARY_NAME} ${MAIN_SOURCES}
                                  ${CPU_SOURCES}
                                  ${GPU_SOURCES})
# add_library(${LIBRARY_NAME} ${MAIN_SOURCES}
#                ${CPU_SOURCES}
#                ${GPU_SOURCES})
## 2. Link...
target_link_libraries( ${LIBRARY_NAME} ${NVTX_LIBRARY}
                                       ${CUDA_CUDART_LIBRARY}
                                       ${CPP_LIBRARY}
                     )
# target_link_libraries( ${LIBRARY_NAME} ${CPP_LIBRARY}
#                      )

# Link against the NVTX library (optional)
find_library( NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )
