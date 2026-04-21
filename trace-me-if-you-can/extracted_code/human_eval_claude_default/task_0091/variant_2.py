import re
pattern = r'I\s[^.?!]*[.?!]'
matches = re.findall(pattern, S)
return len(matches)
