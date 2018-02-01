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

#include <cstdio>

#include "../flac_base.h"
#include "../flac_type.h"
#include "../flac_const.h"

#include "gpuraku_cuda_common.cup"

extern "C"
{
#include "gpuraku_common.h"
#include "flac_cuda.h"
}

// ---------- Constants ---------- 
__constant__ unsigned char  cudaFlacCrc8Table[256];
__constant__ gruint32       cudaBlockSize[16];
__constant__ gruint32       cudaSampleRate[16];
__constant__ gruint8        cudaChannel[16];
__constant__ gruint8        cudaBps[8];
__constant__ gruint8        cudaChannelAss[16];

#if __CUDA_ARCH__ >= 350
    // For Eric's GTX1080
    #define CudaThreadBlockSize 128
    #define cudaLdg(x)          __ldg(&x)
    #define cudaRLdg(x)         __ldg(x)
#else
    // For my local GTX680M
    #define CudaThreadBlockSize 32
    #define cudaLdg(x)          (x)
    #define cudaRLdg(x)         (*(x))
#endif

#include "flac_cuda_bitstream.cup"

typedef struct CudaFrameDecode
{
    CudaBitStream bitStream;
    gruint32 blockSize;
    grint32 *channelPcm[2];
    gruint8 channel;
    gruint8 channelAssignment;
    gruint8 bitsPerSample;
} CudaFrameDecode;

typedef struct CudaSubFrameType
{
    grint32 coefs[32];
    grint32 *pcm;
    gruint8 order;
    gruint8 type;
    gruint8 shift;
} CudaSubFrameType;

typedef struct GRFlacCuda
{
    size_t *          cudaFrameSizes;  // GPU data of frame sizes.
    uchar *           cudaData;        // GPU data of the raw FLAC data.
    grint32 *         cudaPcm;         // GPU data of the PCM samples.
    CudaSubFrameType *cudaSubFrames;   // GPU decode data.
    CudaFrameDecode  *cudaFrameDecode; // GPU decode data
} GRFlacCuda;

//----Parts----
#include "flac_cuda_frame.cup"
#include "flac_cuda_subframe.cup"
#include "flac_cuda_channel.cup"


void flac_cuda_deploy_constants()
{
    //Prepare the constant tables.
    cudaMemcpyToSymbol(cudaFlacCrc8Table,   flacCrc8Table, 
                        sizeof(flacCrc8Table));
    cudaMemcpyToSymbol(cudaBlockSize,       flacBlockSize,
                        sizeof(flacBlockSize));
    cudaMemcpyToSymbol(cudaSampleRate,      flacSampleRate, 
                        sizeof(flacSampleRate));
    cudaMemcpyToSymbol(cudaChannel,         flacChannel, 
                        sizeof(flacChannel));
    cudaMemcpyToSymbol(cudaBps,             flacBitsPerSample, 
                        sizeof(flacBitsPerSample));
    cudaMemcpyToSymbol(cudaChannelAss,      flacChannelAssignment, 
                        sizeof(flacChannelAssignment));
}

int flac_cuda_deploy_data(GRFlacDecodeUser *flacUser)
{   
    GRFlacCuda *flacCuda=NULL;
    //Reset the pointer to NULL first.
    flacUser->flacCuda=NULL;
    //Allocate memory for the cuda memory.
    if(!gr_malloc((void **)&flacCuda, sizeof(GRFlacCuda)))
    {
        //Failed to allocate memory.
        return 0;
    }
    //Allocate the memory on graphics card.
    if(!gr_cuda_malloc(flacCuda->cudaData, flacUser->frameDataSize))
    {
        //Roll back.
        gr_free((void **)&flacCuda);
        //Failed to allocate memory.
        return 0;
    }
    if(!gr_cuda_malloc(flacCuda->cudaFrameSizes, 
                       flacUser->frameSizeLength))
    {
        //Roll back.
        cudaFree(flacCuda->cudaData);
        gr_free((void **)&flacCuda);
        //Failed to allocate memory.
        return 0;
    }
    if(!gr_cuda_malloc(flacCuda->cudaFrameDecode, 
                       flacUser->frameCount*sizeof(CudaFrameDecode)))
    {
        //Roll back.
        cudaFree(flacCuda->cudaData);
        cudaFree(flacCuda->cudaFrameSizes);
        gr_free((void **)&flacCuda);
        //Failed to allocate memory.
        return 0;
    }
    if(!gr_cuda_malloc(flacCuda->cudaSubFrames, 
                       flacUser->frameCount*sizeof(CudaSubFrameType)))
    {
        //Roll back.
        cudaFree(flacCuda->cudaData);
        cudaFree(flacCuda->cudaFrameSizes);
        cudaFree(flacCuda->cudaFrameDecode);
        gr_free((void **)&flacCuda);
        //Failed to allocate memory.
        return 0;
    }
    if(!gr_cuda_malloc(flacCuda->cudaPcm, 
                       flacUser->pcmSize))
    {
        //Roll back.
        cudaFree(flacCuda->cudaData);
        cudaFree(flacCuda->cudaFrameSizes);
        cudaFree(flacCuda->cudaFrameDecode);
        cudaFree(flacCuda->cudaSubFrames);
        gr_free((void **)&flacCuda);
        //Failed to allocate memory.
        return 0;
    }
    //Copy the data to the device.
    gr_cuda_memcpy_to_device(
               flacCuda->cudaData, 
               flacUser->firstFrame, 
               flacUser->frameDataSize);
    //Set the data to the pointer.
    flacUser->flacCuda=flacCuda;
    return 1;
}

void flac_cuda_decode(GRFlacDecodeUser *flacUser,
                      size_t *frameSizes,
                      grint32 *pcm)
{
    //Get the flac cuda pointer.
    GRFlacCuda *flacCuda=flacUser->flacCuda;
    //Allocate the start position.
    gruint64 blockCount=(flacUser->frameCount+CudaThreadBlockSize-1)/CudaThreadBlockSize;
    //Find all the frames.
    //Decode the frame header, initialized the bit stream to the start position 
    //of the sub frame.
    flac_cuda_find_frames<<<blockCount, CudaThreadBlockSize>>>(
        flacCuda->cudaData, 
        flacUser->frameDataSize,
        flacCuda->cudaFrameSizes,
        flacCuda->cudaFrameDecode,
        flacUser->lastPos,
        flacUser->searchSize,
        flacUser->frameCount, 
        flacUser->frameCount, 
        flacUser->streamInfo.sampleRate, 
        flacUser->streamInfo.channels, 
        flacUser->streamInfo.bitsPerSample);
    //gr_cuda_memcpy_to_host(
    //    frameSizes, flacCuda->cudaFramePos, flacUser->frameSizeLength);
    //for(size_t i=0; i<flacUser->frameCount; ++i)
    //{
    //    printf("%lu\t%lu\n", i, frameSizes[i]);
    //}
    //Loop, decode sub frame for each channel.
    for(gruint8 i=0; i<flacUser->streamInfo.channels; ++i)
    {
        flac_cuda_decode_sub_frame<<<blockCount, CudaThreadBlockSize>>>(
            flacCuda->cudaFrameDecode, 
            i, 
            flacUser->frameLength, 
            flacUser->streamInfo.maximumBlockSize, 
            flacUser->frameCount,
            flacCuda->cudaSubFrames, 
            flacCuda->cudaPcm);
        flac_cuda_restore_signal<<<blockCount, CudaThreadBlockSize>>>(
            flacCuda->cudaFrameDecode, 
            flacCuda->cudaSubFrames, 
            flacUser->frameCount);
    }
    //Interchannel decorrelation.
    flac_cuda_decorrelate_interchannel<<<blockCount, CudaThreadBlockSize>>>(
        flacCuda->cudaFrameDecode, 
        flacUser->frameCount);

    //Copy the data back.
    gr_cuda_memcpy_to_host(
        frameSizes, flacCuda->cudaFrameSizes, flacUser->frameSizeLength);
    gr_cuda_memcpy_to_host(
        pcm, flacCuda->cudaPcm, flacUser->pcmSize);
}

void flac_cuda_free_data(GRFlacDecodeUser *flacUser)
{
    //Get the flac cuda pointer.
    GRFlacCuda *flacCuda=flacUser->flacCuda;
    //Free all the data.
    cudaFree(flacCuda->cudaData);
    cudaFree(flacCuda->cudaFrameSizes);
    cudaFree(flacCuda->cudaFrameDecode);
    cudaFree(flacCuda->cudaSubFrames);
    cudaFree(flacCuda->cudaPcm);
    free(flacCuda);
    //Reset the pointer.
    flacUser->flacCuda=NULL;
}