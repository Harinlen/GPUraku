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

#ifndef GPURAKU_ENCODER
#define GPURAKU_ENCODER

/*
 * This file contains all the data that is needed to describe an encoder.
 */

#include "gpuraku_base.h"

typedef struct GRPcmData GRPcmData;

typedef struct GREncoder
{
    /*
     * The name of the encoding format by the decoder, all in lower case.
     */
    const char *name;

    /*
     * Encode the data into byte array.
     */
    void (*encode)(GRPcmData *pcmData, uchar **data, size_t *size);
} GREncoder;

typedef struct GREncodeData
{
    // The encoder which would be used.
    GREncoder *encoder;
    // The PCM data for the encoder.
    GRPcmData *pcmData;
} GREncodeData;

#endif // GPURAKU_ENCODER