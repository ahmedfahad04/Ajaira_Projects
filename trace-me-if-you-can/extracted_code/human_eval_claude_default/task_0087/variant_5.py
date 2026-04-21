# Version 5: Using generator with custom sort key function
def get_sort_key(coordinate):
    return (coordinate[0], -coordinate[1])

def find_coordinates():
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            if lst[i][j] == x:
                yield (i, j)

return sorted(find_coordinates(), key=get_sort_key)
