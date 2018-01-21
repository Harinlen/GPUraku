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

#ifndef GPURAKU
#define GPURAKU

/*
 * This is the only header files that for outer projects to use. All the 
 * necessary functions and structures will be exposed here. So no other things 
 * need to include to use this library.
 */

typedef struct GRInputFile GRInputFile;
typedef struct GRDecodeData GRDecodeData;
typedef struct GREncodeData GREncodeData;
typedef struct GRPcmData GRPcmData;

/*
 * Usage: gr_init_all_decoders();
 * Initialize all the decoders of GPUraku.
 * Returns: 1       - All the decoders are loaded successfully.
 *          0       - At least one decoder cannot be loaded.
 */
int gr_init_all_decoders();

/*
 * Usage: gr_init_all_encoders();
 * Initialize all the encoders of GPUraku.
 * Returns: 1       - All the encoders are loaded successfully.
 *          0       - At least one encoder cannot be loaded.
 */
int gr_init_all_encoders();

/*
 * Usage: gr_init_all_codecs();
 * Initialize all the decoders and encoders
 * Returns: 1		- All the codecs are loaded successfully.
 * 			0		- At least one codec cannot be loaded.
 */
int gr_init_all_codecs();

/*
 * Usage: GRByteArray *fileData=gr_open_input_file("/usr/a.flac");
 * Load a file content and map the data to the memory space. The file content 
 * will be mapped into a structure. It must be closed after open.
 * Params:  filename- The path to the file.
 * Returns: NULL    - Failed to open the file or allocate the memory.
 *          (other) - The file content has been read to the byte array at the 
 *                    return value pointer.
 */
GRInputFile *gr_open_input_file(const char *filename);

/*
 * Usage: gr_close_input_file(&fileData);
 * Close an opened input file, free the memory and system resources.
 * Params:  file    - The pointer to input file structure pointer.
 */
void gr_close_input_file(GRInputFile **file);

/*
 * Usage: int result = gr_write_to_file("/usr/a.wav", data, 256);
 * Write binary data to a specific file path.
 * Params:  filename- The path to the file.
 *          data    - The raw data pointer.
 *          size    - The size of the data.
 * Returns: 1       - Successfully write the data to file.
 *          0       - Failed to write data to file.
 */
int gr_write_to_file(const char *filename, 
                     const unsigned char *data, size_t size);

/*
 * Usage: int result = gr_find_decoder(fileData, &decoderContext);
 * Try to find a decoder for the file, and load the decoding data to the 
 * decoding context.
 * Params:  fileData- The loaded input file structure pointer.
 *          data    - The decoding data prepared for the decoding, including the
 *                    decoder and its support data.
 * Returns: 0       - Failed to find a decoder for the file data.
 *          1       - Successfully find a decoder. The decoding data will be 
 *                    written into the context.
 */
int gr_find_decoder(GRInputFile *fileData, GRDecodeData **data);

/*
 * Usage: int result = gr_allocate_pcm_data(data, &pcmData);
 * Allocate the memory for the PCM data according to the decoder data, prepare 
 * the decode data for the decoding.
 * Params:  data    - The decoder context data pointer.
 *          pcmData - The pointer to the PCM data structure pointer.
 * Returns: 0       - Failed to allocate memory for the PCM data.
 *          1       - Successfully allocate memory for the PCM data.
 */
int gr_allocate_pcm_data(GRDecodeData *data, GRPcmData **pcmData);

/*
 * Usage: gr_decode(decoderContext, pcmData);
 * Decode the entire file into raw PCM data.
 * Params:  fileData- The decoder context of a specific file.
 *          pcmData - The pointer to a PCM data structure.
 */
void gr_decode(GRDecodeData *data, GRPcmData *pcmData);

/*
 * Usage: int result = gr_find_encoder("wav", pcmData, &encoderContext);
 * Try to find the encoder for the file, and allocate memory for the encode 
 * context.
 * Params:  format  - The name of the format.
 *          pcmData - The decodec PCM data.
 *          data    - The pointer to the encoded data.
 * Returns: 0       - Failed to find the encoder or cannot prepared for audio 
 *                    encoding.
 *          1       - Successfully find the encoder and allocated encode data.
 */
int gr_find_encoder(const char *format, GRPcmData *pcmData, 
                    GREncodeData **data);

/*
 * Usage: gr_encode(pcmData, "wav", &wavRawData, &wavRawSize);
 * Dump a PCM data into a specific format file. The parameter of the encoding 
 * will use the parameter in the PCM data structure. 
 * Params:  data    - The pointer to a PCM structure.
 *          rawData - The unsigned char array for saving data.
 *          size    - The pointer to the size for char array size.
 * Returns: 0       - Failed to dump the data to the memory.
 *          1       - Successfully dump the data to the memory.
 */
void gr_encode(GREncodeData *data, unsigned char **rawData, size_t *size);

/*
 * Usage: gr_free_decode_data(&decoderContext);
 * Free the decoder data.
 * Params:  data    - The pointer to the decoder data structure pointer.
 */
void gr_free_decode_data(GRDecodeData **data);

/*
 * Usage: gr_free_pcm(&pcmData);
 * Free the pcm data.
 * Params:  pcmData - The pointer to the PCM data structure pointer.
 */
void gr_free_pcm(GRPcmData **pcmData);

#endif // GPURAKU