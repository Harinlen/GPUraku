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

#ifndef FLAC_TYPE
#define FLAC_TYPE

#include <stddef.h>

/*
 * Define the internal types and tables of the FLAC formats.
 */

#include "gpuraku_base.h"

typedef struct FlacStreamInfo
{
    gruint16    minimumBlockSize;   // <16>
    gruint16    maximumBlockSize;   // <16>
    gruint32    minimumFrameSize;   // <24>
    gruint32    maximumFrameSize;   // <24>
    gruint32    sampleRate;         // <20> maximum 655350
    gruint8     channels;           // <3> channels-1, range 1~8.
    gruint8     bitsPerSample;      // <5> bps-1, range 4-32.
    gruint64    totalSamples;       // <36>
} FlacStreamInfo;

typedef struct GRFlacCuda GRFlacCuda;

typedef struct GRFlacDecodeUser
{
    FlacStreamInfo  streamInfo;      // FLAC stream info.
    gruint32        frameCount;      // Entire frame counts.
    size_t          frameSizeLength; // Size of frame length array.
    size_t          frameLength;     // Length of one frame.
    size_t          pcmSize;         // Size of PCM sample array.
    size_t          searchSize;      // Size of each thread search.
    size_t          frameDataSize;   // Size of frame contents.
    size_t          lastPos;         // The position of the last frame.
    uchar *         firstFrame;      // The pointer to the first frame.
    GRFlacCuda *    flacCuda;        // The cuda accelerate data.
} GRFlacDecodeUser;

#endif // FLAC_TYPE