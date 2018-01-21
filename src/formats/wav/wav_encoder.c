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

#include "gpuraku_type.h"
#include "gpuraku_common.h"

#include "wav_encoder.h"

#define GR_WRITE_U16LE(data, value) \
    data[0]= value & 0x00FF      ; \
    data[1]=(value & 0xFF00) >> 8; \
    data += 2

#define GR_WRITE_U24LE(data, value) \
    data[0]= value & 0x0000FF       ; \
    data[1]=(value & 0x00FF00) >>  8; \
    data[2]=(value & 0xFF0000) >> 16; \
    data += 3

#define GR_WRITE_U32LE(data, value) \
    data[0]= value & 0x000000FF       ; \
    data[1]=(value & 0x0000FF00) >>  8; \
    data[2]=(value & 0x00FF0000) >> 16; \
    data[3]=(value & 0xFF000000) >> 24; \
    data += 4

#define GRAudioPcmSample(wav, frameIndex, frameSize, channel, channelSize, sampleIndex) \
    wav->pcm[frameIndex * frameSize + channel * channelSize + sampleIndex]

typedef void (*WavWriteSample)(uchar **rawData, grint32 value);

void gr_write_s8(uchar **rawData, grint32 value)
{
    (**rawData)=(grint8)value;
    ++(*rawData);
}

void gr_write_s16le(uchar **rawData, grint32 value)
{
    uchar *data=(*rawData);
    GR_WRITE_U16LE(data, value);
    (*rawData)=data;
}

void gr_write_s24le(uchar **rawData, grint32 value)
{
    uchar *data=(*rawData);
    GR_WRITE_U24LE(data, value);
    (*rawData)=data;
}

void gr_write_s32le(uchar **rawData, grint32 value)
{
    uchar *data=(*rawData);
    GR_WRITE_U32LE(data, value);
    (*rawData)=data;
}

static char wavFmtChunkId[]={0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00};

static inline void wav_write_header(
    GRPcmData *pcm,
    gruint32 dataSize,
    uchar **rawData)
{
    uchar *data=(*rawData);
    gruint32 riffSize=dataSize+36;
    gruint32 sampleChannelSize=pcm->channels*pcm->bitPerSample,
             byteRate=(pcm->sampleRate*sampleChannelSize)>>3;
    gruint16 blockAlign=sampleChannelSize>>3;
    // Write the RIFF chunk descriptor..
    gr_write_data(&data, "RIFF", 4);
    GR_WRITE_U32LE(data, riffSize);
    gr_write_data(&data, "WAVE", 4);
    // - write the fmt subchunk.
    gr_write_data(&data, wavFmtChunkId, 8);
    //   * Audio format.
    GR_WRITE_U16LE(data, 0x0001);
    //   * Number of Channels.
    GR_WRITE_U16LE(data, pcm->channels);
    //   * Sample Rate.
    GR_WRITE_U32LE(data, pcm->sampleRate);
    //   * Byte Rate.
    GR_WRITE_U32LE(data, byteRate);
    //   * Block Align.
    GR_WRITE_U16LE(data, blockAlign);
    //   * Bits per sample.
    GR_WRITE_U16LE(data, pcm->bitPerSample);
    // - data subchunk
    gr_write_data(&data, "data", 4);
    GR_WRITE_U32LE(data, dataSize);
    // Set the data back.
    (*rawData)=data;
}

void wav_encode(GRPcmData *pcmData, uchar **data, size_t *size)
{
    //Allocate memory for the data.
    size_t fileSize=32, dataChunkSize=0, sampleByte, frame, sampleIndex, 
           channel, frameSize=pcmData->channels*pcmData->maxFrameSize;
    //Sum up the frame count first.
    for(size_t i=0; i<pcmData->frameCount; ++i)
    {
        //Get the frame count.
        dataChunkSize+=pcmData->frameSize[i];
    }
    //Sum up and calculate the file size.
    sampleByte=(pcmData->bitPerSample+7)>>3;
    dataChunkSize *= sampleByte * pcmData->channels;
    fileSize+=dataChunkSize;
    //Write the sample to the data by frame order.
    WavWriteSample sampleWrite=NULL;
    switch(sampleByte)
    {
    case 1:
        sampleWrite=gr_write_s8;
        break;
    case 2:
        sampleWrite=gr_write_s16le;
        break;
    case 3:
        sampleWrite=gr_write_s24le;
        break;
    case 4:
        sampleWrite=gr_write_s32le;
        break;
    default:
        return;
    }
    //Allocate memory for the raw data.
    uchar *wavRawData=NULL;
    if(!gr_malloc((void **)&wavRawData, fileSize))
    {
        //Failed to allocate memory for the wav raw data.
        return;
    }
    //Set the data.
    (*data)=wavRawData;
    (*size)=fileSize;
    uchar *wavData=wavRawData;
    //Write the WAV file header.
    wav_write_header(pcmData, dataChunkSize, &wavData);
    ////Write the sample to the data by frame order.
    for(frame=0; frame<pcmData->frameCount; ++frame)
    {
        //Loop for the frame size.
        for(sampleIndex=0; 
            sampleIndex<pcmData->frameSize[frame]; 
            ++sampleIndex)
        {
            //Write the channel one by one.
            for(channel=0; channel<pcmData->channels; ++channel)
            {
                //Write the data.
                (*sampleWrite)(&wavData, 
                               GRAudioPcmSample(pcmData, 
                                                frame, frameSize, channel, 
                                                pcmData->maxFrameSize, 
                                                sampleIndex));
            }
        }
    }
}