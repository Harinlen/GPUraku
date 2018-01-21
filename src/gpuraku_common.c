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

#include "gpuraku_base.h"

#include "gpuraku_common.h"

BOOL gr_malloc(void **target, size_t size)
{
    //Use malloc function to allocate the memory.
    void *buffer=malloc(size);
    //Check the allocate result.
    if(buffer)
    {
        //Set the memory to the target.
        (*target)=buffer;
        //Complete.
        return TRUE;
    }
    //Failed to allocate memory.
    return FALSE;
}

GRInputFile *gr_create_input_file()
{
    GRInputFile *inputFile = NULL;
    // Allocate the memory.
    if(!gr_malloc((void **)&inputFile, sizeof(GRInputFile)))
    {
        // Failed to allocate the memory.
        return NULL;
    }
    // Simply reset the value of a byte array.
    inputFile->size = 0;
    inputFile->file = -1;
    inputFile->data = NULL;
    return inputFile;
}

void gr_free(void **target)
{
    //Check whether the pointer is NULL.
    if(*target)
    {
        //Free the target.
        free(*target);
        //Reset the pointer to NULL.
        (*target)=NULL;
    }
}