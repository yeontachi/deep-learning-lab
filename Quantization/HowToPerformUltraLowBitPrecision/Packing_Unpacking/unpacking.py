'''
Code block for unpacking bits for ultra-low-bit precision quantization.
'''

def low_bit_dequantize(quantized_tensor, bits, scale, zero_pt):
    num_values = quantized_tensor.shape[1]*bits // 8
    num_steps = 8 // bits 
    mask = 2 ** bits - 1 # [0000 1111]
    
    dequantized_tensor = torch.zeros((quantized_tensor.shape[0], quantized_tensor.shape[1]*num_steps),
                                      dtype = torch.uint8)
    
    for row_num in range(quantized_tensor.shape[0]):
        row = quantized_tensor[row_num, :] 
        dequantized_row = torch.zeros((dequantized_tensor.shape[1]), dtype=torch.uint8) # [0, 0, 0, 0]
        unpacked_idx = 0
        for i in range(quantized_tensor.shape[1]):
            for j in range(num_steps):
                dequantized_row[unpacked_idx] |= row[i] >> (bits * j) 
                unpacked_idx += 1
                
        dequantized_row &= mask
        dequantized_tensor[row_num, :] = dequantized_row
        
    dequantized_tensor = dequantize(dequantized_tensor, scale, zero_pt)
    
    return dequantized_tensor
