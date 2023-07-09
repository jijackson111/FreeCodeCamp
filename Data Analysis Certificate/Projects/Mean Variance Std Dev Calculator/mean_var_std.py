import numpy as np

def calculate(list):
    # Create 3x3 array
    arr = np.array(list).reshape(3, 3)
    
    # Mean
    m1 = np.mean(arr, axis=0)
    m2 = np.mean(arr, axis=1)
    m3 = np.mean(arr)
    
    # Variance
    v1 = np.var(arr, axis=0)
    v2 = np.var(arr, axis=1)
    v3 = np.var(arr)
    
    # Standard deviation
    s1 = np.std(arr, axis=0)
    s2 = np.std(arr, axis=1)
    s3 = np.std(arr)
    
    # Max
    ma1 = np.max(arr, axis=0)
    ma2 = np.max(arr, axis=1)
    ma3 = np.max(arr)
    
    # Min
    mi1 = np.min(arr, axis=0)
    mi2 = np.min(arr, axis=1)
    mi3 = np.min(arr)
    
    # Sum
    su1 = np.sum(arr, axis=0)
    su2 = np.sum(arr, axis=1)
    su3 = np.sum(arr)
    
    # Dictionary
    dictionary = {'mean': [m1, m2, m3],
                  'variance': [v1, v2, v3],
                  'standard deviation': [s1, s2, s3],
                  'max': [ma1, ma2, ma3],
                  'min': [mi1, mi2, mi3],
                  'sum': [su1, su2, su3]}
    
    return dictionary

print(calculate([2,6,2,8,4,0,1,5,7]))
    