from collections import defaultdict

   def find_max_freq(lst):
       freq_count = defaultdict(int)
       for num in lst:
           freq_count[num] += 1

       max_freq = -1
       for i, freq in enumerate(freq_count.values()):
           if freq >= i:
               max_freq = i

       return max_freq
