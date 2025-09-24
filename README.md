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
