def refactor_v4(s):
    import re
    def sort_word(match):
        return ''.join(sorted(match.group()))
    return re.sub(r'\S+', sort_word, s)
