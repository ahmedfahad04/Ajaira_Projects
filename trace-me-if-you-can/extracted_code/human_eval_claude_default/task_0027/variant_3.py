import string as str_module
translation_table = str.maketrans(
    str_module.ascii_lowercase + str_module.ascii_uppercase,
    str_module.ascii_uppercase + str_module.ascii_lowercase
)
return string.translate(translation_table)
