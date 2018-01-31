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

// This is the channel assignment codes for GTX1080

__global__ void flac_cuda_left_assignment(
    grint32 *leftChannel, 
    grint32 *rightChannel)
{
    //The right channel is not the original data.
    rightChannel[threadIdx.x]=leftChannel[threadIdx.x]-rightChannel[threadIdx.x];
}

__global__ void flac_cuda_right_assignment(
    grint32 *leftChannel, 
    grint32 *rightChannel)
{
    //The right channel is not the original data.
    leftChannel[threadIdx.x]+=rightChannel[threadIdx.x];
}

__global__ void flac_cuda_mid_assignment(
    grint32 *leftChannel, 
    grint32 *rightChannel)
{
    //The right channel is not the original data.
    grint32 side=rightChannel[threadIdx.x], 
            mid=(leftChannel[threadIdx.x]<<1) | (side & 1);
    leftChannel[threadIdx.x]=(mid+side)>>1;
    rightChannel[threadIdx.x]=(mid-side)>>1;
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
    if(channelAssignment==FLAC_CHANNEL_INDEPENDENT)
    {
        //No need to do any thing.
        return;
    }
    //Quick calculate the block size.
    gruint32 blockSize=decodeData[threadId].blockSize;
    //Get the channel pointer.
    grint32 *left=decodeData[threadId].channelPcm[0],
            *right=decodeData[threadId].channelPcm[1];
    if(channelAssignment==FLAC_CHANNEL_LEFT_ASSIGNMENT)
    {
        flac_cuda_left_assignment<<<1, blockSize>>>(left, right);
        return;
    }
    if(channelAssignment==FLAC_CHANNEL_MID_ASSIGNMENT)
    {
        flac_cuda_mid_assignment<<<1, blockSize>>>(left, right);
        return;
    }
    //FLAC_CHANNEL_RIGHT_ASSIGNMENT
    flac_cuda_right_assignment<<<1, blockSize>>>(left, right);
}