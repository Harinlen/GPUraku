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
 
## Set the project name
project(GPUraku)
# Defines the cmake required
cmake_minimum_required(VERSION 2.8)

# Add the GPURaku library
add_subdirectory(src)
# Add the example file of the library
add_subdirectory(examples)