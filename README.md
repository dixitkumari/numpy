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
