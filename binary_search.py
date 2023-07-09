# Searching function
def binary_search(list1, target):
    first = 0
    last = len(list1) - 1
    
    while first <= last:
        midpoint = (first + last) // 2
        
        if list1[midpoint] == target:
            return midpoint
        elif list1[midpoint] < target:
            first = midpoint + 1
        else:
            last = midpoint - 1
            
    return None

# Verification function
def verify(index):
    if index is not None:
        print("Target found at index:", index)
    else:
        print("Target not found in list")
        
# Test algorithm
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = binary_search(numbers, 9)
verify(result)       