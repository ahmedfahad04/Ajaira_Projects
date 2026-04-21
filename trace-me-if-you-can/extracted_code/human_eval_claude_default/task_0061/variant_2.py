counter = 0
def check_balance():
    nonlocal counter
    for b in brackets:
        counter += 1 if b == "(" else -1
        if counter < 0:
            return True
    return False

return not check_balance() and counter == 0
