# GPUraku Drei Mk.X
Copyright (C) 2017-2018 Haolei Ye and Eric McCreath

## Overview
GPUraku is a CUDA powered audio decoding framework.

## Dependencies
* CMake (>=2.8)
* GCC (>=5.0)
* nvcc (>=8.0)

## Compile
To compile GPUraku, ensure you have installed all the dependencies, then execute

    cmake -DCMAKE_BUILD_TYPE=Release .

after that, execute

    make

A simple example has been built under `bin` directory.

## License

GPUraku is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version.

GPUraku is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with GPUraku. If not, see http://www.gnu.org/licenses/.

For more information, please see LICENSES.

