# Counter Function
from collections import Counter
a = 'aaabbccc'
my_counter = Counter(a)
print('\nThe string was used to create a character frequency dictionary:\n', my_counter)

# namedtuple Function
from collections import namedtuple
Point = namedtuple('Point', 'x, y')
pt = Point(1, -4)
print('\nA graph point was created:\n', pt)

# OrderedDict Function
from collections import OrderedDict
ordered_dict = OrderedDict()
ordered_dict['b'] = 2
ordered_dict['c'] = 3
ordered_dict['d'] = 4
ordered_dict['a'] = 1
print('\nUnordered input order did not affect the indexes:\n', ordered_dict)

# defaultdict Function
from collections import defaultdict
d = defaultdict(int)
d['a'] = 1
d['b'] = 2
d['c']
print('\nDictionary with a default value if not given one:\n', d)

# deque Function
from collections import deque
de = deque()
de.append(1)
de.append(2)
de.appendleft([3, 4, 5])
print('\nDouble ended queue:\n', de)