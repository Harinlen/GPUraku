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
#include <stdlib.h>
#include <string.h>

#include "gpuraku_type.h"
#include "gpuraku_common.h"

#include "flac_base.h"
#include "flac_type.h"
#include "flac_const.h"

#include "cuda/flac_cuda.h"

#include "flac_decoder.h"

const char flac_header[4] = {0x66, 0x4C, 0x61, 0x43};

static inline BOOL flac_crc8(uchar *data, int len)
{
    //Initialize the CRC instance data.
    gruint8 crc=0;
    //Loop for entire length.
    while(len--)
    {
        //Check the table, update the value.
        crc=flacCrc8Table[crc ^ *data++];
    }
    //Check whether the CRC-8 position is correct.
    return crc==(*data);
}

static inline int flac_utf8_length(uchar *frame)
{
    return (frame[0]==0xFE) ? 7 :
            (((frame[0] & 0xFE)==0xFC) ? 6 :
             (((frame[0] & 0xFC)==0xF8) ? 5 :
             (((frame[0] & 0xF8)==0xF0) ? 4 :
             (((frame[0] & 0xF0)==0xE0) ? 3 : 
             (((frame[0] & 0xE0)==0xC0) ? 2 : 1)))));
}

static inline gruint64 flac_utf8(uchar *pos)
{
    return (((pos[0] & 0x80)==0) ?
        ((gruint64)pos[0]) :
    (((pos[0] & 0xE0) == 0xC0) ?
        ((((gruint64)pos[0] & 0x1F) << 6)  | (pos[1] & 0x3F)):
    (((pos[0] & 0xF0) == 0xE0) ?
        ((((gruint64)pos[0] & 0x0F) << 12) | ((pos[1] & 0x3F) << 6) |
          (pos[2] & 0x3F)):
    (((pos[0] & 0xF8) == 0xF0) ?
        ((((gruint64)pos[0] & 0x07) << 18) | ((pos[1] & 0x3F) << 12) |
         ((pos[2] & 0x3F) << 6 ) | (pos[3] & 0x3F)):
    (((pos[0] & 0xFC) == 0xF8) ?
        ((((gruint64)pos[0] & 0x03) << 24) | ((pos[1] & 0x3F) << 18) |
         ((pos[2] & 0x3F) << 12) | ((pos[3] & 0x3F) << 6 ) |
          (pos[4] & 0x3F)):
    (((pos[0] & 0xFE) == 0xFC) ?
        ((((gruint64)pos[0] & 0x01) << 30) | ((pos[1] & 0x3F) << 24) |
         ((pos[2] & 0x3F) << 18) | ((pos[3] & 0x3F) << 12) |
         ((pos[4] & 0x3F) << 6 ) |  (pos[5] & 0x3F)):
    ((pos[0] == 0xFE) ?
        ((((gruint64)pos[1] & 0x3F) << 30) | ((pos[2] & 0x3F) << 24) |
         ((pos[3] & 0x3F) << 18) | ((pos[4] & 0x3F) << 12) |
         ((pos[5] & 0x3F) << 6 ) |  (pos[6] & 0x3F)):
        (0))))))));
}

static inline gruint8 flac_is_frame_header(uchar *frame, gruint8 streamChannels)
{
    //Check the frame variable is valid.
    //   Block size           Sample rate
    if ((frame[2] & 0xF0) && ((frame[2] & 0x0F)!=0x0F) &&
                 // Channel                  Bit per sample
                ((frame[3] & 0xF0)<0xB0) && ((frame[3] & 0x0E)!=0x06) &&
                 // Bit per sample 
                ((frame[3] & 0x0E)!=0x0E) &&
                 // Stream channel validation. 
                (flacChannel[(frame[3] & 0xF0)>>4]<=streamChannels))
    {
        //Calculate the entire frame header size.
        gruint8 headerLength=4+flac_utf8_length(frame+4)+
                (((frame[2] & 0xF0)==0x60)?1:(((frame[2] & 0xF0)==0x70)?2:0))+
                (((frame[2] & 0x0F)==0x0C)?1:(((frame[2] & 0x0D)>0x0C)?2:0));
        //Execute the CRC-8 checking.
        if(flac_crc8(frame, headerLength))
        {
            //If success, return the header length.
            return headerLength;
        }
    }
    // Or else, return 0.
    return 0;
}

BOOL flac_check(GRInputFile *file, void **user)
{
    uchar *data=file->data;
    //Check the first four bytes.
    if(memcmp(data, flac_header, 4) ||
        //Allocate the user structure.
        !gr_malloc(user, sizeof(GRFlacDecodeUser)))
    {
        // It is not a FLAC file or failed to allocate memory.
        return FALSE;
    }
    //Recast the flac user
    GRFlacDecodeUser *flacUser=(GRFlacDecodeUser *)*user;
    //Read the STREAMINFO metadata block.
    uchar *flacData=file->data+8, *flacEnd=file->data+file->size;
    FlacStreamInfo *streamInfo=&(flacUser->streamInfo);
    gruint64 cache64;
    gruint32 blockSize;
    streamInfo->minimumBlockSize=TO_UINT16BE(flacData);
    streamInfo->maximumBlockSize=TO_UINT16BE(flacData);
    streamInfo->minimumFrameSize=TO_UINT24BE(flacData);
    streamInfo->maximumFrameSize=TO_UINT24BE(flacData);
    cache64=TO_UINT64BE(flacData);
    streamInfo->sampleRate      = (cache64 & 0xFFFFF00000000000) >> 44;
    streamInfo->channels        =((cache64 & 0x00000E0000000000) >> 41)+1;
    streamInfo->bitsPerSample   =((cache64 & 0x000001F000000000) >> 36)+1;
    streamInfo->totalSamples    = (cache64 & 0x0000000FFFFFFFFF);
    flacData+=16;
    //Skip the other blocks, jump to frames.
    while(!(flacData[0] & 0x80) && flacData < flacEnd)
    {
        ++flacData;
        //Read the block size.
        blockSize=TO_UINT24BE(flacData);
        flacData+=blockSize;
    }
    //Read the last block size and skip it.
    ++flacData;
    blockSize=TO_UINT24BE(flacData);
    flacData+=blockSize;
    //Now, the flac data is at the position of the frame start.
    //Find the last frame position.
    while(flacEnd > flacData)
    {
        //Check the sync code and reserved 0.
        if(flacEnd[0]==0xFF && (flacEnd[1] & 0xFE)==0xF8 &&
                (flacEnd[3] & 0x01)==0x00 &&
                //Execute the FLAC frame checking. 
                flac_is_frame_header(flacEnd, streamInfo->channels))
        {
            break;
        }
        //Move forward
        --flacEnd;
    }
    //Calculate the last frame index.
    flacUser->firstFrame=flacData;
    flacUser->lastPos=flacEnd-flacData;
    flacUser->frameDataSize=file->size-(flacData-file->data);
    flacUser->frameCount=flac_utf8(flacEnd+4)+1;
    //Calculate the sizes of the flac user.
    flacUser->frameSizeLength=flacUser->frameCount*sizeof(size_t);
    flacUser->frameLength=streamInfo->maximumBlockSize*streamInfo->channels;
    flacUser->pcmSize=
        flacUser->frameLength*flacUser->frameCount*sizeof(grint32);
    flacUser->searchSize=flacUser->frameDataSize/flacUser->frameCount;
    flacUser->flacCuda=NULL;
    //Deploy constant data to graphics card.
    flac_cuda_deploy_constants();
    //Complete stream info fetch.
    return TRUE;
}

BOOL flac_allocate_pcm(void *user, GRPcmData **pcmData)
{
    // Allocate memory for the pcm data.
    GRPcmData *wavData=NULL;
    if(!gr_malloc((void **)&wavData, sizeof(GRPcmData)))
    {
        //Failed to allocate basic struct memory.
        return FALSE;
    }
    //Recast the FLAC user data.
    GRFlacDecodeUser *flacUser=(GRFlacDecodeUser *)user;
    //Allocate the PCM data.
    if(!gr_malloc((void **)&(wavData->frameSize), flacUser->frameSizeLength))
    {
        //Roll back.
        free(wavData);
        //Failed to allocate PCM frame size length array.
        return FALSE;
    }
    if(!gr_malloc((void **)&(wavData->pcm), flacUser->pcmSize))
    {
        //Roll back.
        free(wavData->frameSize);
        free(wavData);
        //Failed to allocate PCM sample array.
        return FALSE;
    }
    //Allocate the CUDA memory.
    if(!flac_cuda_deploy_data(flacUser))
    {
        //Failed to allocate memory on the graphics card.
        //Roll back.
        free(wavData->pcm);
        free(wavData->frameSize);
        free(wavData);
        //Failed.
        return FALSE;
    }
    //Get the stream info.
    FlacStreamInfo *streamInfo=&(flacUser->streamInfo);
    //Set the stream info data to PCM data.
    wavData->sampleRate=streamInfo->sampleRate;
    wavData->maxFrameSize=streamInfo->maximumBlockSize;
    wavData->bitPerSample=streamInfo->bitsPerSample;
    wavData->channels=streamInfo->channels;
    wavData->frameCount=flacUser->frameCount;
    //Set the pointer to the pcm data.
    (*pcmData)=wavData;
    return TRUE;
}

void flac_decode(void *user, GRPcmData *pcmData)
{
    //Call the CUDA function to decode all the data.
    flac_cuda_decode((GRFlacDecodeUser *)user,
                     pcmData->frameSize,
                     pcmData->pcm);
}

void flac_free_user(void **user)
{
    //Recast the flac user
    GRFlacDecodeUser *flacUser=(GRFlacDecodeUser *)*user;
    //Check whether the cuda is freed.
    if(flacUser->flacCuda)
    {
        //Free the cuda data.
        flac_cuda_free_data(flacUser);
    }
    //Free the pointer data and reset the pointer to NULL.
    gr_free(user);
}