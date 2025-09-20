'''
Code block for packing bits for ultra-low-bit precision quantization.
'''
import torch 
import torch.nn.functional as F 

def low_bit_quantize(input_tensor, bits):
    assert(input_tensor.shape[0]*bits)%8 == 0
    
    num_cols_needed = input_tensor.shape[1]*bits // 8 
    num_shifts = 8 // bits 
    
    final_quantized_tensor = torch.zeros((input_tensor.shape[0],
                                          num_cols_needed),
                                         dtype=torch.uint8)
    
    for row_num in range(input_tensor.shape[0]):
        row = input_tensor[row_num, :] # [0, 9, 11, 15]
        quantized_row = torch.zeros((num_cols_needed), dtype=torch.uint8)
        unpacked_idx = 0
        for i in range(num_cols_needed):
            for j in range(num_shifts):
                quantized_row[i] |= row[unpacked_idx] << (bits * j)
                unnpacked_idx += 1
                
                
        final_quantized_tensor[row_num, :] = quantized_row