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

#ifndef FLAC_BASE
#define FLAC_BASE

#include "gpuraku_base.h"

/*
 * This file is designed for storing all the constants value of the FLAC format.
 */

// Metadata block type
#define FLAC_BLOCK_STREAMINFO           0
#define FLAC_BLOCK_PADDING              1
#define FLAC_BLOCK_APPLICATION          2
#define FLAC_BLOCK_SEEKTABLE            3
#define FLAC_BLOCK_VORBISCOMMENT        4
#define FLAC_BLOCK_CUESHEET             5
#define FLAC_BLOCK_PICTURE              6
#define FLAC_BLOCK_INVALID              127

// Channel assignments
#define FLAC_CHANNEL_INDEPENDENT        0
#define FLAC_CHANNEL_LEFT_ASSIGNMENT    1
#define FLAC_CHANNEL_RIGHT_ASSIGNMENT   2
#define FLAC_CHANNEL_MID_ASSIGNMENT     3
#define FLAC_CHANNEL_RESERVED           0xFF

//Sub frame types.
#define FLAC_SUBFRAME_CONSTANT          0
#define FLAC_SUBFRAME_VERBATIM          1
#define FLAC_SUBFRAME_FIXED             2
#define FLAC_SUBFRAME_LPC               3

#define TO_UINT16BE(x) (x[0]<<8 | x[1]); x+=2
#define TO_UINT24BE(x) (x[0]<<16 | x[1]<<8 | x[2]); x+=3
#define TO_UINT64BE(x) ((gruint64)x[0]<<56 | (gruint64)x[1]<<48 \
                    | (gruint64)x[2]<<40 | (gruint64)x[3]<<32 \
                    | (gruint64)x[4]<<24 | (gruint64)x[5]<<16 \
                    | (gruint64)x[6]<<8  | (gruint64)x[7]); x+=8

#endif // FLAC_BASE