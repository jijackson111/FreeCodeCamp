# Searching function
def linear_search(list1, target):
    for i in range(0, len(list1)):
        if list1[i] == target:
            return i
    return None

# Verification function
def verify(index):
    if index is not None:
        print("Target found at index:", index)
    else:
        print("Target not found in list")
        
# Test algorithm
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = linear_search(numbers, 9)
verify(result)
