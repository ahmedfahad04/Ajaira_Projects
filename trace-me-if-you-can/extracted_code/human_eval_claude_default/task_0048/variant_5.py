text_len = len(text)
for i, char in enumerate(text):
    if i >= text_len // 2:
        break
    if char != text[text_len - 1 - i]:
        return False
return True
