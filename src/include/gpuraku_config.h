/*
 * This file is part of GPUraku.
 * 
 * GPUraku is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * GPUraku is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GPUraku.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef GPURAKU_CONFIG
#define GPURAKU_CONFIG

/*
 * Defines the configuration of the GPUraku library.
 */

#define GR_VERSION_MAJOR    10
#define GR_VERSION_MINOR    0
#define GR_VERSION_PATCH    0
#define GR_VERSION_APPEND   ""

//----DON'T TOUCH BELOW---
//Generate string for version.
//MAGIC, DON'T TOUCH
// Stringify \a x.
#define _TOSTR(x)   #x
// Stringify \a x, perform macro expansion.
#define TOSTR(x)  _TOSTR(x)

/* the following are compile time version */
/* C++11 requires a space between literal and identifier */
#define GR_VERSION_STR \
    TOSTR(GR_VERSION_MAJOR) "." TOSTR(GR_VERSION_MINOR) "." \
    TOSTR(GR_VERSION_PATCH) " " GR_VERSION_APPEND
//----DON'T TOUCH ABOVE---

#endif // GPURAKU_CONFIG