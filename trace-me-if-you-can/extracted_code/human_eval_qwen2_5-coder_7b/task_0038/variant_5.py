def encode_twice(str_input):
    return encode_cyclic(encode_cyclic(str_input))

final_result = encode_twice(s)
return final_result
