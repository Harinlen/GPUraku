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

#ifndef FLAC_DECODER
#define FLAC_DECODER

#include "gpuraku_decoder.h"

// All the folloing functions are internal functions which matches the port 
// define of the decoder.
BOOL flac_check(GRInputFile *file, void **user);
BOOL flac_allocate_pcm(void *user, GRPcmData **pcmData);
void flac_decode(void *user, GRPcmData *pcmData);
void flac_free_user(void **user);

static GRDecoder flac_decoder = {
    .check        = flac_check,
    .allocate_pcm = flac_allocate_pcm,
    .decode       = flac_decode,
    .free_user    = flac_free_user,
};

#endif // FLAC_DECODER
