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

#ifndef GPURAKU_BASE
#define GPURAKU_BASE

/*
 * This file is designed for storing all the base types of the GPUraku library.
 */

//  Booleans for C
#define BOOL                int
#define FALSE               0
#define TRUE                1

// Size-dependent types (architechture-dependent byte order)
typedef signed char         grint8;     /* 8 bit signed */
typedef unsigned char       gruint8;    /* 8 bit unsigned */
typedef short               grint16;    /* 16 bit signed */
typedef unsigned short      gruint16;   /* 16 bit unsigned */
typedef int                 grint32;    /* 32 bit signed */
typedef unsigned int        gruint32;   /* 32 bit unsigned */
typedef long long           grint64;    /* 64 bit signed */
typedef unsigned long long  gruint64;   /* 64 bit unsigned */

typedef grint64             grlonglong; /* long long */
typedef gruint64            grulonglong;/* long long unsigned */

typedef double              grreal;     /* real */

//Some other types which is useful.
typedef unsigned char       uchar;
typedef unsigned short      ushort;
typedef unsigned int        uint;
typedef unsigned long       ulong;

// Avoid "unused parameter" warnings
#define GR_UNUSED(x) (void)x;

#endif // GPURAKU_BASE