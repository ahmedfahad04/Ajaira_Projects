import re
sentences = re.split(r'[.?!]\s*', S)
return len([s for s in sentences if s.startswith('I ')])
