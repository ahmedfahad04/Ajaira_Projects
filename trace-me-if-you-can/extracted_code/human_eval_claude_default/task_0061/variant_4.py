s = brackets
while "()" in s:
    s = s.replace("()", "")
return s == ""
