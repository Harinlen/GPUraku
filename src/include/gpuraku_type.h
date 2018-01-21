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

#ifndef GPURAKU_TYPE
#define GPURAKU_TYPE

/*
 * Define the internal types of the GPUraku library.
 */

#include <stddef.h>

#include "gpuraku_base.h"

/*
 * GRFile structure defines the access for a read only file.
 */
typedef struct GRInputFile
{
    int    file;        // The file descriptor.
    size_t size;        // Input file size.
    uchar *data;        // Input file data.
} GRInputFile;

typedef struct GRPcmData
{
    gruint32 sampleRate;
    gruint32 maxFrameSize;
    gruint32 frameCount;
    gruint8  bitPerSample;
    gruint8  channels;
    size_t * frameSize;
    grint32 *pcm;
} GRPcmData;

#endif // GPURAKU_TYPE