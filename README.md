# numpy
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from IPython.display import Image

list1 = [10,20,30,40,50,60]
list1

[10, 20, 30, 40, 50, 60]

type(list1)
list


arr1 = np.array(list1)
arr1

array([10, 20, 30, 40, 50, 60])


arr1.data
<memory at 0x000001C2B747E348>


type(arr1)
numpy.ndarray
array([10, 10, 10, 10, 10])
# Generate array of Odd numbers
ar1 = np.arange(1,20)
ar1[ar1%2 ==1]
array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19])
# Generate array of even numbers
ar1 = np.arange(1,20)
ar1[ar1%2 == 0]
array([ 2,  4,  6,  8, 10, 12, 14, 16, 18])
# Generate evenly spaced 4 numbers between 10 to 20.
np.linspace(10,20,4)
array([10.        , 13.33333333, 16.66666667, 20.        ])
# Generate evenly spaced 11 numbers between 10 to 20.
np.linspace(10,20,11)
array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.])
# Create an array of random values
np.random.random(4)
array([0.61387161, 0.7734601 , 0.48868515, 0.05535259])
# Generate an array of Random Integer numbers
np.random.randint(0,500,5)
array([359,   3, 200, 437, 400])
# Generate an array of Random Integer numbers
np.random.randint(0,500,10)
array([402, 196, 481, 426, 245,  19, 292, 233, 399, 175])
# Using random.seed we can generate same number of Random numbers
np.random.seed(123)
np.random.randint(0,100,10)
array([66, 92, 98, 17, 83, 57, 86, 97, 96, 47])
# Using random.seed we can generate same number of Random numbers
np.random.seed(123)
np.random.randint(0,100,10)
array([66, 92, 98, 17, 83, 57, 86, 97, 96, 47])
# Using random.seed we can generate same number of Random numbers
np.random.seed(101)
np.random.randint(0,100,10)
array([95, 11, 81, 70, 63, 87, 75,  9, 77, 40])

# Generate array of Random float numbers
f1 = np.random.uniform(5,10, size=(10))
f1
array([6.5348311 , 9.4680654 , 8.60771931, 5.94969477, 7.77113796,
       6.76065977, 5.90946201, 8.92800881, 9.82741611, 6.16176831])
# Extract Integer part
np.floor(f1)
array([6., 9., 8., 5., 7., 6., 5., 8., 9., 6.])
# Truncate decimal part
np.trunc(f1)
array([6., 9., 8., 5., 7., 6., 5., 8., 9., 6.])
# Convert Float Array to Integer array
f1.astype(int)
array([6, 9, 8, 5, 7, 6, 5, 8, 9, 6])
# Normal distribution (mean=0 and variance=1)
b2 =np.random.randn(10)
b2
array([ 0.18869531, -0.75887206, -0.93323722,  0.95505651,  0.19079432,
        1.97875732,  2.60596728,  0.68350889,  0.30266545,  1.69372293])
arr1
array([10, 20, 30, 40, 50, 60])
# Enumerate for Numpy Arrays
for index, value in np.ndenumerate(arr1):
    print(index, value)
(0,) 10
(1,) 20
(2,) 30
(3,) 40
(4,) 50
(5,) 60


**Operations on an Array**
arr2 = np.arange(1,20)
arr2
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19])
# Sum of all elements in an array
arr2.sum()
190
# Cumulative Sum
np.cumsum(arr2)
array([  1,   3,   6,  10,  15,  21,  28,  36,  45,  55,  66,  78,  91,
       105, 120, 136, 153, 171, 190], dtype=int32)
# Find Minimum number in an array
arr2.min()
1
# Find MAX number in an array
arr2.max()
19
# Find INDEX of Minimum number in an array
arr2.argmin()
0
# Find INDEX of MAX number in an array
arr2.argmax()
18
# Find mean of all numbers in an array
arr2.mean()
10.0
# Find median of all numbers present in arr2
np.median(arr2)
10.0
# Variance
np.var(arr2)
30.0
# Standard deviation
np.std(arr2)
5.477225575051661
# Calculating percentiles
np.percentile(arr2,70)
13.6
# 10th & 70th percentile
np.percentile(arr2,[10,70])
array([ 2.8, 13.6])


**Operations on a 2D Array**
A = np.array([[1,2,3,0] , [5,6,7,22] , [10 , 11 , 1 ,13] , [14,15,16,3]])
A
array([[ 1,  2,  3,  0],
       [ 5,  6,  7, 22],
       [10, 11,  1, 13],
       [14, 15, 16,  3]])
# SUM of all numbers in a 2D array
A.sum()
129
# MAX number in a 2D array
A.max()
22
# Minimum
A.min()
0
# Column wise mimimum value 
np.amin(A, axis=0)
array([1, 2, 1, 0])
# Row wise mimimum value 
np.amin(A, axis=1)
array([0, 5, 1, 3])
# Mean of all numbers in a 2D array
A.mean()
8.0625
# Mean
np.mean(A)
8.0625
# Median
np.median(A)
6.5
# 50 percentile = Median
np.percentile(A,50)
6.5
np.var(A)
40.30859375
np.std(A)
6.348904925260734
np.percentile(arr2,70)
13.6
# Enumerate for Numpy 2D Arrays
for index, value in np.ndenumerate(A):
    print(index, value)
(0, 0) 1
(0, 1) 2
(0, 2) 3
(0, 3) 0
(1, 0) 5
(1, 1) 6
(1, 2) 7
(1, 3) 22
(2, 0) 10
(2, 1) 11
(2, 2) 1
(2, 3) 13
(3, 0) 14
(3, 1) 15
(3, 2) 16
(3, 3) 3

**Reading elements of an array**
a = np.array([7,5,3,9,0,2])
# Access first element of the array
a[0]
7
# Access all elements of Array except first one.
a[1:]
array([5, 3, 9, 0, 2])
# Fetch 2nd , 3rd & 4th value from the Array
a[1:4]
array([5, 3, 9])
# Get last element of the array
a[-1]
2
a[-3]
9
a[-6]
7
a[-3:-1]
array([9, 0])


**Replace elements in array**
ar = np.arange(1,20)
ar
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19])
# Replace EVEN numbers with ZERO
rep1 = np.where(ar % 2 == 0, 0 , ar)
print(rep1)
[ 1  0  3  0  5  0  7  0  9  0 11  0 13  0 15  0 17  0 19]
ar2 = np.array([10, 20 , 30 , 10 ,10 ,20, 20])
ar2
array([10, 20, 30, 10, 10, 20, 20])
# Replace 10 with value 99
rep2 = np.where(ar2 == 10, 99 , ar2)
print(rep2)
[99 20 30 99 99 20 20]
p2 = np.arange(0,100,10)
p2
array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
# Replace values at INDEX loc 0,3,5 with 33,55,99
np.put(p2, [0, 3 , 5], [33, 55, 99])
p2
array([33, 10, 20, 55, 40, 99, 60, 70, 80, 90])



**Missing Values in an array**
a = np.array([10 ,np.nan,20,30,60,np.nan,90,np.inf])
a
array([10., nan, 20., 30., 60., nan, 90., inf])
# Search for missing values and return as a boolean array
np.isnan(a)
array([False,  True, False, False, False,  True, False, False])
# Index of missing values in an array
np.where(np.isnan(a))
(array([1, 5], dtype=int64),)
# Replace all missing values with 99
a[np.isnan(a)] = 99
a
array([10., 99., 20., 30., 60., 99., 90., inf])
# Check if array has any NULL value
np.isnan(a).any()
False
A = np.array([[1,2,np.nan,4] , [np.nan,6,7,8] , [10 , np.nan , 12 ,13] , [14,15,16,17]])
A
array([[ 1.,  2., nan,  4.],
       [nan,  6.,  7.,  8.],
       [10., nan, 12., 13.],
       [14., 15., 16., 17.]])
# Search for missing values and return as a boolean array
np.isnan(A)
array([[False, False,  True, False],
       [ True, False, False, False],
       [False,  True, False, False],
       [False, False, False, False]])
# Index of missing values in an array
np.where(np.isnan(A))
(array([0, 1, 2], dtype=int64), array([2, 0, 1], dtype=int64))


Stack Arrays Vertically
a = np.zeros(20).reshape(2,-1)
b = np.repeat(1, 20).reshape(2,-1)
a
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
b
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
np.vstack([a,b])
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
a1 = np.array([[1], [2], [3]])
b1 = np.array([[4], [5], [6]])
a1
array([[1],
       [2],
       [3]])
b1
array([[4],
       [5],
       [6]])
np.vstack([a1,b1])
array([[1],
       [2],
       [3],
       [4],
       [5],
       [6]])
