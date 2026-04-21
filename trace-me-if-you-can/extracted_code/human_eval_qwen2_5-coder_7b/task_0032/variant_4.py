start, finish = -1.0, 1.0
    while poly(xs, start) * poly(xs, finish) > 0:
        start *= 2.0
        finish *= 2.0
    while (finish - start) > 1e-10:
        middle = (start + finish) / 2.0
        if poly(xs, middle) * poly(xs, start) > 0:
            start = middle
        else:
            finish = middle
    return start
