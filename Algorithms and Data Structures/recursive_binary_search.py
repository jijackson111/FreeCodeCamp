# Searching function
def recursive_binary_search(list1, target):
    if len(list1) == 0:
        return False
    else:
        midpoint = (len(list1)) // 2
        
        if list1[midpoint] == target:
            return True
        else:
            if list1[midpoint] < target:
                return recursive_binary_search(list1[midpoint+1:], target)
            else:
                return recursive_binary_search(list1[:midpoint], target)
            
# Verification function
def verify(result):
    print("Target found:", result)
    
# Test algorithm
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = recursive_binary_search(numbers, 9)
verify(result)      
