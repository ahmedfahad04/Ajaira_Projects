def twice_encode(input_str):
    return encode_cyclic(encode_cyclic(input_str))

output = twice_encode(s)
return output
