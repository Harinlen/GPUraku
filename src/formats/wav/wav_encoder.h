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

#ifndef WAV_ENCODER
#define WAV_ENCODER

#include "gpuraku_encoder.h"

void wav_encode(GRPcmData *pcmData, uchar **data, size_t *size);

static GREncoder wav_encoder = {
    .name   = "wav",
    .encode = wav_encode,
};

#endif // WAV_ENCODER