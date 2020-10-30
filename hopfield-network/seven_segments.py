from math import *
import numpy as np

def seven_segment(pattern):

    def to_bool(a):
        if a==1:
            return True
        return False
    

    def hor(d):
        if d:
            print(" _ ")
        else:
            print("   ")
    
    def vert(d1,d2,d3):
        word=""

        if d1:
            word="|"
        else:
            word=" "
        
        if d3:
            word+="_"
        else:
            word+=" "
        
        if d2:
            word+="|"
        else:
            word+=" "
        
        print(word)

    

    pattern_b=list(map(to_bool,pattern))

    hor(pattern_b[0])
    vert(pattern_b[1],pattern_b[2],pattern_b[3])
    vert(pattern_b[4],pattern_b[5],pattern_b[6])

    number=0
    for i in range(0,4):
        if pattern_b[7+i]:
            number+=pow(2,i)
    print(int(number))

six=[1,1,-1,1,1,1,1,-1,1,1,-1]
three=[1,-1,1,1,-1,1,1,1,1,-1,-1]
one=[-1,-1,1,-1,-1,1,-1,1,-1,-1,-1]

print("reference patterns:")
seven_segment(three)
seven_segment(six)
seven_segment(one)
print("---")

# create the matrix of connection weights W
weight_matrix = np.zeros((11, 11))

# put the patterns in a list so we can iterate over them
patterns = [six, three, one]

# we want to iterate over every unique (i,j) pair
for i in range(0, 11):
    for j in range(i+1, 11):
        # compute the weight value at this point
        v = 0
        for pattern in patterns:
            v += pattern[i] * pattern[j]
        v /= len(patterns)

        # set the appropriate values in the matrix
        weight_matrix[i, j] = v
        weight_matrix[j, i] = v

# define a function to iterate a pattern according
# to the McCulloch-Pitts relation
def evolve(pattern):
    # create an array in which to store the new pattern
    p = []

    # compute each value based on the original pattern
    for i in range(len(pattern)):
        v = 0
        for j in range(len(pattern)):
            v += weight_matrix[i][j] * pattern[j]
        if v > 0:
            v = 1
        else:
            v = -1
        p.append(v)

    # check if we have converged
    for i in range(len(p)):
        if p[i] != pattern[i]:
            return p # HAVE NOT CONVERGED
    
    # otherwise return none if we have converged
    return None

# create a function to keep evolving a pattern until it converges
def converge(pattern):
    p = pattern
    print("original: ", p)
    while True:
        candidate = evolve(p)
        if candidate == None:
            print("converged:", p)
            return p
        p = candidate
        print("step:     ", p)

print("\ntest1")

test=[1,-1,1,1,-1,1,1,-1,-1,-1,-1]
seven_segment(test)
test=converge(test)
seven_segment(test)

print("\ntest2")

test=[1,1,1,1,1,1,1,-1,-1,-1,-1]
seven_segment(test)
test=converge(test)
seven_segment(test)
