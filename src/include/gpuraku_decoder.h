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

#ifndef GPURAKU_DECODER
#define GPURAKU_DECODER

/*
 * This file contains all the data that is needed to describe a decoder.
 */

#include "gpuraku_base.h"

typedef struct GRInputFile GRInputFile;

typedef struct GRDecoder
{
    /*
     * Check the type of the file could be decode by the decoder or not.
     * If the file content cannot be decoded, return FALSE.
     * If the file content could be decoded, prepared the decode data, like 
     * parsing headers and some other data for decoding, return TRUE.
     */
    BOOL (*check)(GRInputFile *file, void **user);

    /*
     * Allocate the PCM data according to the audio data.
     */
    BOOL (*allocate_pcm)(void *user, GRPcmData **pcmData);

    /*
     * Decode the encoded audio data to PCM data.
     */
    void (*decode)(void *user, GRPcmData *pcmData);

    /*
     * Free the user context data.
     */
    void (*free_user)(void **user);
} GRDecoder;

typedef struct GRDecodeData
{
    // The decoder which would be used as the file.
    GRDecoder *decoder;
    // The user data of the decoder.
    void *user;
} GRDecodeData;

#endif // GPURAKU_DECODER