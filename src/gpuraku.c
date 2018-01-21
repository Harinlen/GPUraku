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
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "gpuraku_type.h"
#include "gpuraku_common.h"

#include "gpuraku.h"

GRInputFile *gr_open_input_file(const char *filename)
{
	//Allocate the byte array memory.
	struct stat fileStat;
    int result;
    GRInputFile *inputFile=gr_create_input_file();
    //Check the result.
    if(!inputFile)
    {
        //Failed to allocate the file data.
        return NULL;
    }
    //Open the file content.
    inputFile->file=open(filename, O_RDONLY, 0666);
    if(inputFile->file==-1)
    {
        free(inputFile);
        //Failed to open the file as binary read only mode.
        return NULL;
    }
    //Get the size of the file.
    //Get the file state.
    result=fstat(inputFile->file, &fileStat);
    if(result==-1)
    {
        free(inputFile);
        //Failed to get file size.
        return NULL;
    }
    //Save the file size.
    inputFile->size=fileStat.st_size;
    //Map the file data to the memory.
    inputFile->data=(uchar *)mmap(NULL, 
                                  fileStat.st_size, 
                                  PROT_READ, 
                                  MAP_SHARED, 
                                  inputFile->file, 
                                  0);
    //Give back the structure.
    return inputFile;
}

int gr_write_to_file(const char *filename, const uchar *data, size_t size)
{
    //Open the target file as binary write.
    FILE *outputFile=fopen(filename, "wb");
    //Write the data.
    int result=(fwrite(data, sizeof(uchar), size, outputFile)==size);
    //Close the file.
    fclose(outputFile);
    //Give back the result.
    return result;
}

void gr_close_input_file(GRInputFile **file)
{
    //Get the input file strcucture.
    GRInputFile *inputFile=(*file);
    //Unmapped the data of the file.
    munmap((void *)inputFile->data, inputFile->size);
    //Free the memory.
    free(inputFile);
    //Reset the pointer.
    (*file)=NULL;
}

void gr_free_pcm(GRPcmData **pcmData)
{
    //Get the PCM data structure.
    GRPcmData *data=(*pcmData);
    //Free the array data.
    free(data->frameSize);
    free(data->pcm);
    //Free the memory.
    free(data);
    //Reset the pointer.
    (*pcmData)=NULL;
}