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

#ifndef GPURAKU_COMMON
#define GPURAKU_COMMON

/*
 * Defines the functions that only depends on the standard C library.
 */

#include <string.h>

#include "gpuraku_base.h"
#include "gpuraku_type.h"

/*
 * Usage: gpuraku_malloc(&file, sizeof(GRFileContent));
 * Allocate memory for a structure or something, and then check the result.
 * Params:  target  - The pointer to the data pointer.
 *          size    - The size of the memory in bytes.
 * Returns: TRUE    - The memory has been successfully allocated.
 *          FALSE   - There is no enough memory, allocate failed.
 * NOTE: the target memory won't be freed before allocated new memory. This 
 * might caused memory leaking.
 * If the memory allocate failed, target won't be touched.
 */
BOOL gr_malloc(void **target, size_t size);

/*
 * Usage: GRInputFile *file = gr_create_input_file();
 * Create a input file structure in the memory.
 * Returns: NULL    - Failed to allocate the memory for the input file.
 *          (other) - The pointer to the input file structure. All parameters 
 *                    are set to make the input file an invalid structure.
 */
GRInputFile *gr_create_input_file();

/*
 * Usage: gr_write_data(&data, "your data", 9*sizeof(char));
 * Write content to the data pointer, and move the data pointer to the end.
 * Params:  dest    - The pointer to the data array pointer.
 *          src     - The source data.
 *          size    - The length of the source data.
 */
static inline void gr_write_data(uchar **dest, const char *src, size_t size)
{
    //Copy the data from the source.
    memcpy(*dest, src, size);
    //Move the destination data.
    (*dest)+=size;
}

/*
 * Usage: gr_free(&file);
 * Free a pointer with its data by the default free function and reset the data
 * pointer to NULL.
 * Params:  The pointer to the data pointer.
 */
void gr_free(void **target);

#endif // GPURAKU_COMMON