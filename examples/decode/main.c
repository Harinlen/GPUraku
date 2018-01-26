/*
 * This file is part of GPUraku examples.
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

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/timeb.h>

#include "gpuraku.h"

#define EXIT_FAILURE 1
#define EXIT_SUCCESS 0

#define print_usage(exeName) \
    printf("Usage: %s Decoding file [Dumped file]\n", exeName);
#define print_error_message(errorInfo, exeName) \
    printf("\n%s: %s\n", exeName, errorInfo);

static inline long long getSystemTime()
{
    struct timeb t;
    ftime(&t);
    return 1000 * t.time + t.millitm;
}

// Print for the error information.
void exit_error_usage(const char *errorInfo, char *exeName)
{
    print_error_message(errorInfo, exeName);
    print_usage(exeName);
}

int main(int argc, char *argv[])
{
    // Check the parameter.
    if(argc!=2 && argc!=3)
    {
        // Invalid parameter found.
        exit_error_usage("incorrect parameter found", argv[0]);
        exit(EXIT_FAILURE);
    }
    // Initial all the codecs.
    if(!gr_init_all_codecs())
    {
        //Cannot initial the decoder.
        print_error_message("failed to initialize codecs", argv[0]);
        exit(EXIT_FAILURE);
    }
    // Load the file content.
    GRDecodeData *decoderContext=NULL;
    GRInputFile *inputFile=gr_open_input_file(argv[1]);
    if(!inputFile)
    {
        //Failed to load the file.
        print_error_message("failed to load input file:", argv[0]);
        printf("%s\n", argv[1]);
        exit(EXIT_FAILURE);
    }
    //Find the decoder for the file.
    printf("Finding the decoder...");
    // Find the decoder for the input file.
    if(!gr_find_decoder(inputFile, &decoderContext))
    {
        // Failed to find the decoder.
        print_error_message("failed to find the decoder", argv[0]);
        gr_free_decode_data(&decoderContext);
        gr_close_input_file(&inputFile);
        exit(EXIT_FAILURE);
    }
    printf("done\n");
    // Prepare the pointer for the PCM audio data.
    GRPcmData *pcmData=NULL;
    // Allocate the memory for PCM.
    printf("Allocating memory...");
    if(!gr_allocate_pcm_data(decoderContext, &pcmData))
    {
        printf("\n");
        //Failed to allocate PCM data.
        print_error_message("failed to allocate PCM data memory", argv[0]);
        // Clean the memory.
        gr_free_decode_data(&decoderContext);
        gr_close_input_file(&inputFile);
        exit(EXIT_FAILURE);
    }
    printf("done\n");
    // Start to decode the data.
    printf("Start decoding...");
    long long endTime, startTime=getSystemTime();
    gr_decode(decoderContext, pcmData);
    endTime=getSystemTime();
    printf("done\nDecoding time usage: %lld ms\n", endTime-startTime);
    // Clean the memory.
    gr_free_decode_data(&decoderContext);
    gr_close_input_file(&inputFile);
    //Check for WAV dump option.
    if(argc==3)
    {
        printf("Dump WAV...");
        //We need to dump the WAV.
        GREncodeData *encoderContext=NULL;
        if(!gr_find_encoder("wav", pcmData, &encoderContext))
        {
            //Failed to find encoder.
            print_error_message("failed to find wav encoder", argv[0]);
            //Clean the memory.
            gr_free_pcm(&pcmData);
            exit(EXIT_FAILURE);
        }
        //Start to encode WAV.
        unsigned char *rawData=NULL;
        size_t rawDataSize;
        gr_encode(encoderContext, &rawData, &rawDataSize);
        //Write the raw data to the file.
        gr_write_to_file(argv[2], rawData, rawDataSize);
        printf("done\n");
        //Free the raw data.
        free(rawData);
    }
    //Free the PCM data.
    gr_free_pcm(&pcmData);
    return EXIT_SUCCESS;
}