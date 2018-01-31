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

// This is the channel assignment codes for 3.0

__device__ void flac_cuda_left_assignment(
    grint32 * __restrict__ leftChannel, 
    grint32 * __restrict__ rightChannel, 
    gruint32 blockSize)
{
    //The right channel is not the original data.
    for(gruint32 i=0; i<blockSize; ++i)
    {
        rightChannel[i]=leftChannel[i]-rightChannel[i];
    }
}

__device__ void flac_cuda_right_assignment(
    grint32 * __restrict__ leftChannel, 
    grint32 * __restrict__ rightChannel, 
    gruint32 blockSize)
{
    //The left channel is not the original data.
    for(gruint32 i=0; i<blockSize; ++i)
    {
        leftChannel[i]+=rightChannel[i];
    }
}

__device__ void flac_cuda_mid_assignment(
    grint32 * __restrict__ leftChannel, 
    grint32 * __restrict__ rightChannel, 
    gruint32 blockSize)
{
    //The left channel is not the original data.
    grint32 mid, side;
    for(gruint32 i=0; i<blockSize; ++i)
    {
        side=rightChannel[i];
        mid= (leftChannel[i]<<1) | (side & 1);
        leftChannel[i]=(mid+side)>>1;
        rightChannel[i]=(mid-side)>>1;
    }
}


__global__ void flac_cuda_decorrelate_interchannel(
    CudaFrameDecode *decodeData,
    gruint32 frameCount)
{
    //Calculate the core index.
    int threadId=blockDim.x * blockIdx.x + threadIdx.x;
    if(threadId >= frameCount)
    {
        return;
    }
    //Get the sub frame type.
    gruint8 channelAssignment=decodeData[threadId].channelAssignment;
    gruint32 blockSize=decodeData[threadId].blockSize;
    if(channelAssignment==FLAC_CHANNEL_INDEPENDENT)
    {
        //No need to do any thing.
        return;
    }
    //Get the channel pointer.
    grint32 *left=decodeData[threadId].channelPcm[0],
            *right=decodeData[threadId].channelPcm[1];
    if(channelAssignment==FLAC_CHANNEL_LEFT_ASSIGNMENT)
    {
        flac_cuda_left_assignment(left, right, blockSize);
        return;
    }
    if(channelAssignment==FLAC_CHANNEL_MID_ASSIGNMENT)
    {
        flac_cuda_mid_assignment(left, right, blockSize);
        return;
    }
    //FLAC_CHANNEL_RIGHT_ASSIGNMENT
    flac_cuda_right_assignment(left, right, blockSize);
}