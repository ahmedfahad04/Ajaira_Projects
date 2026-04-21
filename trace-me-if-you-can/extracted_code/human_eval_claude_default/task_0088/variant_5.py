return (lambda arr: [] if not arr else 
        sorted(arr, reverse=((arr[0] + arr[-1]) % 2 == 0)))(array)
