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

#include "gpuraku_common.h"
#include "gpuraku_decoder.h"
#include "gpuraku_encoder.h"

// Decoders
#include "formats/flac/flac_decoder.h"
// Encoders
#include "formats/wav/wav_encoder.h"

#include "gpuraku.h"

// The structure to stores the codec.
typedef struct GRCodecNode
{
    union                       // This union save the codec pointer.
    {
        void *codec;            // The codec content.
        GRDecoder *decoder;     // The parse the codec as decoder.
        GREncoder *encoder;     // The parse the codec as encoder.
    };
    struct GRCodecNode *next;   // The next node position.
} GRCodecNode;
// The codec list.
static GRCodecNode *decoders = NULL;
static GRCodecNode *encoders = NULL;

BOOL gr_init_codec(GRCodecNode **node, void *codec)
{
    //Create node for the decoder.
    GRCodecNode *codecNode=NULL;
    if(!gr_malloc((void **)(&codecNode), sizeof(GRCodecNode)))
    {
        //Failed to allocate memory.
        return FALSE;
    }
    //Set the decoder node data.
    codecNode->codec = codec;
    codecNode->next = NULL;
    //Append the decoder node to end of list.
    while(*node!=NULL)
    {
        //Pass the next node to current.
        node = &((*node)->next);
    }
    //Set the current node to the end of the node.
    (*node) = codecNode;
    return TRUE;
}

#define INIT_DECODER(codec) \
    if(!gr_init_codec(&decoders, &codec)) return 0;

#define INIT_ENCODER(codec) \
    if(!gr_init_codec(&encoders, &codec)) return 0;

int gr_init_all_decoders()
{
    //Check the decoder list initialized state.
    if(decoders)
    {
        //The decoder list is already initialized.
        return 1;
    }
    //Append the decoders to the list.
    INIT_DECODER(flac_decoder);
    // Complete.
    return 1;
}

int gr_init_all_encoders()
{
    //Check the encoder list initialized state.
    if(encoders)
    {
        //The encoder list is already initialized.
        return 1;
    }
    //Append the encoder to the list.
    INIT_ENCODER(wav_encoder);
    // Complete.
    return 1;
}

int gr_init_all_codecs()
{
    //Initialize all the decoders and encoders.
    return gr_init_all_decoders() && gr_init_all_encoders();
}

void gr_free_decode_content(GRDecoder *decoder, void **user)
{
    // Check the decoder context.
    if(decoder->free_user)
    {
        // Free the user function.
        decoder->free_user(user);
        return;
    }
    // Use the default free function.
    free(*user);
    (*user)=NULL;
}

void gr_free_decode_data(GRDecodeData **data)
{
    //Free the decode content.
    gr_free_decode_content((*data)->decoder, &((*data)->user));
    //Free the data itself.
    free(*data);
    (*data)=NULL;
}

int gr_find_decoder(GRInputFile *fileData, GRDecodeData **data)
{
    // Prepare the empty context.
    void *context = NULL;
    GRDecoder *decoder = NULL;
    // Get the first decoder.
    GRCodecNode **node = &decoders;
    // Loop until the end, try to decode the file.
    while((*node)!=NULL)
    {
        //Check the current decoder.
        if((*node)->decoder->check(fileData, &context))
        {
            //We find the decoder to decode the data.
            //Get the decoder.
            decoder=(*node)->decoder;
            //Create the decoder context.
            if(!gr_malloc((void **)data, sizeof(GRDecodeData)))
            {
                // Failed to allocate decode context, free the data.
                // Free the decoder data.
                gr_free_decode_content(decoder, &context);
                return FALSE;
            }
            //Set the decode data.
            (*data)->decoder=decoder;
            (*data)->user=context;
            //Complete.
            return TRUE;
        }
        //Go to next node.
        node = &(*node)->next;
    }
    // Failed to find the data.
    return FALSE;
}

int gr_allocate_pcm_data(GRDecodeData *data, GRPcmData **pcmData)
{
    //Get the decoder.
    GRDecoder *decoder = data->decoder;
    //Call the decoder allocate PCM function.
    return decoder->allocate_pcm(data->user, pcmData);
}

void gr_decode(GRDecodeData *data, GRPcmData *pcmData)
{
    //Get the decoder.
    GRDecoder *decoder=data->decoder;
    //Call the decoder to decode all the PCM.
    decoder->decode(data->user, pcmData);
}

int gr_find_encoder(const char *format, GRPcmData *pcmData, GREncodeData **data)
{
    // Prepare the empty context.
    GREncoder *encoder = NULL;
    // Get the first decoder.
    GRCodecNode **node = &encoders;
    // Loop until the end, try to decode the file.
    while((*node)!=NULL)
    {
        //Check the current decoder.
        if(!strcmp((*node)->encoder->name, format))
        {
            //We find the encoder for encode the format.
            //Get the encoder.
            encoder=(*node)->encoder;
            //Create the decoder context.
            if(!gr_malloc((void **)data, sizeof(GREncodeData)))
            {
                // Failed to allocate encode context, free the data.
                return FALSE;
            }
            //Set the decode data.
            (*data)->encoder=encoder;
            (*data)->pcmData=pcmData;
            //Complete.
            return TRUE;
        }
        //Go to next node.
        node = &(*node)->next;
    }
    // Failed to find the data.
    return FALSE;
}

void gr_encode(GREncodeData *data, unsigned char **rawData, size_t *size)
{
    //Get the encoder.
    GREncoder *encoder=data->encoder;
    //Call the encode function.
    encoder->encode(data->pcmData, rawData, size);
}
