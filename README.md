**numpy-and-pandas-codes-basic-**
The following repo contains basic coding to python libaries i.e, numpy and pandas

**NUMPY BASIC CODES**
# Import Numpy Library
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from IPython.display import Image



list1 = [10,20,30,40,50,60]
list1


# Display the type of an object
type(list1)


# Convert list to Numpy Array
arr1 = np.array(list1)
arr1


# Memory address of an array object
arr1.data



# Display type of an object
type(arr1)


# Datatype of array
arr1.dtype


# Convert Integer Array to FLOAT
arr1.astype(float)


# Generate evenly spaced numbers (space =1) between 0 to 10
np.arange(0,10)


# Generate numbers between 0 to 100 with a space of 10
np.arange(0,100,10)



# Generate numbers between 10 to 100 with a space of 10 in descending order
np.arange(100, 10, -10)



# Shape of Array
arr3 = np.arange(0,10)
arr3.shape



# Size of array
arr3.size


# Bytes consumed by one element of an array object
arr3.itemsize


# Bytes consumed by an array object
arr3.nbytes


# Length of array
len(arr3)


# Generate an array of zeros
np.zeros(10)


# Generate an array of ones with given shape
np.ones(10)



# Repeat 10 five times in an array
np.repeat(10,5)



# Repeat each element in array 'a' thrice
a= np.array([10,20,30])
np.repeat(a,3)


# Generate array of Odd numbers
ar1 = np.arange(1,20)
ar1[ar1%2 ==1]


# Generate array of even numbers
ar1 = np.arange(1,20)
ar1[ar1%2 == 0]



# Create an array of random values
np.random.random(4)


# Generate an array of Random Integer numbers
np.random.randint(0,500,5)



arr2 = np.arange(1,20)
arr2


# Sum of all elements in an array
arr2.sum()

# Cumulative Sum
np.cumsum(arr2)


# Find Minimum number in an array
arr2.min()


#  Find MAX number in an array
arr2.max()


# Find mean of all numbers in an array
arr2.mean()


# Find median of all numbers present in arr2
np.median(arr2)


# Variance
np.var(arr2)

# Standard deviation
np.std(arr2)


# Calculating percentiles
np.percentile(arr2,70)


A = np.array([[1,2,3,0] , [5,6,7,22] , [10 , 11 , 1 ,13] , [14,15,16,3]])
A


# SUM of all numbers in a 2D array
A.sum()

# MAX number in a 2D array
A.max()

# Minimum
A.min()


# Enumerate for Numpy 2D Arrays
for index, value in np.ndenumerate(A):
    print(index, value)



a = np.array([7,5,3,9,0,2])


# Access first element of the array
a[0]

# Access all elements of Array except first one.
a[1:]


# Fetch 2nd , 3rd & 4th value from the Array
a[1:4]


# Get last element of the array
a[-1]


ar = np.arange(1,20)
ar


# Replace EVEN numbers with ZERO
rep1 = np.where(ar % 2 == 0, 0 , ar)
print(rep1)



# Replace 10 with value 99
rep2 = np.where(ar2 == 10, 99 , ar2)
print(rep2)



p2 = np.arange(0,100,10)
p2

# Replace values at INDEX loc 0,3,5 with 33,55,99
np.put(p2, [0, 3 , 5], [33, 55, 99])
p2




a = np.array([10, np.nan,20,30,60,np.nan,90,np.inf])
a


# Search for missing values and return as a boolean array
np.isnan(a)


# Index of missing values in an array
np.where(np.isnan(a))


# Replace all missing values with 99
a[np.isnan(a)] = 99
a


# Check if array has any NULL value
np.isnan(a).any()



# Stack arrays vertically
a = np.zeros(20).reshape(2,-1)
b = np.repeat(1, 20).reshape(2,-1)
a b

np.vstack([a,b])



# Stack arrays horizontally

np.hstack([a,b])
np.hstack([a1,b1])



# Common items between two arrays

c1 = np.array([10,20,30,40,50,60])
c2 = np.array([12,20,33,40,55,60])
np.intersect1d(c1,c2)



# Remove common elements of C1 & C2 array from C1

np.setdiff1d(c1,c2)
