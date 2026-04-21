vowels = set(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'])
return ''.join(filter(lambda x: x not in vowels, text))
