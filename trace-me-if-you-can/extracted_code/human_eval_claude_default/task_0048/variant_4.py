return all(text[i] == text[len(text) - 1 - i] for i in range(len(text) // 2))
