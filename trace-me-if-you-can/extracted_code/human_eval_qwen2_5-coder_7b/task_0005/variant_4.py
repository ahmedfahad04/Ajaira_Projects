def insert_delimiter(data, sep):
    if not data:
        return []

    final_list = []
    for i, item in enumerate(data):
        final_list.append(item)
        if i != len(data) - 1:
            final_list.append(sep)

    return final_list
