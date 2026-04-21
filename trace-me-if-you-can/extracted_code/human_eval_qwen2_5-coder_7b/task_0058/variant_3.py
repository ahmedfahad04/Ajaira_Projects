result = set()
for first_list_item in l1:
    for second_list_item in l2:
        if first_list_item == second_list_item:
            result.add(first_list_item)
return sorted(list(result))
