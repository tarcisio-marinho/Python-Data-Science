
def all_even():
    n = 0
    while True:
        yield n
        n += 2

def evens():
    n = 0
    evens = []
    while True:
        evens.append(n)
        n+=2
    return evens

for i in evens():
    print(i)

for i in all_even():
    print(i)
