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

#ifndef FLAC_CUDA
#define FLAC_CUDA

#include "gpuraku_base.h"

typedef struct GRFlacDecodeUser GRFlacDecodeUser;

/*
 * This file is the only interface of the CUDA accelerating codes. No other 
 * headers should be included.
 */

/*
 * Usage: flac_cuda_deploy_constants();
 * Deploy the constant accelerats data to the graphics card.
 */
void flac_cuda_deploy_constants();

/*
 * Usage: int result = flac_cuda_deploy_data(flacUser);
 * Allocate the global memory for the specific FLAC audio.
 * Params:  flacUser- The FLAC user structure of a specfic audio.
 * Returns: 0       - Failed to allocate memory for the audio.
 *          1       - Successfully allocate graphics memory for the audio.  
 */
int flac_cuda_deploy_data(GRFlacDecodeUser *flacUser);

/*
 * Usage: flac_cuda_decode(flacUser, pcmData->frameSize, pcmData->pcm);
 * Decode all the FLAC encoded data to the pcm raw data.
 */
void flac_cuda_decode(GRFlacDecodeUser *flacUser,
                      size_t *frameSizes,
                      grint32 *pcm);

/*
 * Usage: flac_cuda_free_data(flacUser);
 * Free the cuda allocated data.
 */
void flac_cuda_free_data(GRFlacDecodeUser *flacUser);

#endif // FLAC_CUDA