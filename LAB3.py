import numpy as np
import math
import csv
from sklearn.tree import DecisionTreeClassifier

def read_data(filename):
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        headers = next(datareader)
        metadata = []
        traindata = []
        for name in headers:
            metadata.append(name)
        for row in datareader:
            traindata.append(row)

    return (metadata, traindata)

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""
        
    def __str__(self):
        return self.attribute

def subtables(data, col, delete):

    '''
    Returns items and dict of the data 
    Eg:
    ['Overcast' 'Rainy' 'Sunny']
 {'Overcast': array([[b'Overcast', b'Hot', b'High', b'False', b'Yes'],
       [b'Overcast', b'Cool', b'Normal', b'True', b'Yes'],
       [b'Overcast', b'Mild', b'High', b'True', b'Yes'],
       [b'Overcast', b'Hot', b'Normal', b'False', b'Yes']], dtype='|S32'),
    '''
    dict = {}
    items = np.unique(data[:, col]) #unique of rows Eg [yes, no]
    count = np.zeros((items.shape[0], 1), dtype=np.int32) #init to zeros of items' shape
    
     
    
    for x in range(items.shape[0]): #for every items' row
        for y in range(data.shape[0]): #for every data's row
            if data[y, col] == items[x]: # if data's row == items' value
                count[x] += 1 
                
    for x in range(items.shape[0]):
        dict[items[x]] = np.empty((int(count[x]), data.shape[1]), dtype="|S32") #init to zeros
        print(dict)
        pos = 0
        for y in range(data.shape[0]):
            if data[y, col] == items[x]:
                dict[items[x]][pos] = data[y] #assign keys and values to get the subtable
                pos += 1       
        if delete:
            dict[items[x]] = np.delete(dict[items[x]], col, 1)
    print(f'Subtables ->> {items}')
    print(f'Subtables ->> {dict}')
    return items, dict

def entropy(S):
    print(S)
    items = np.unique(S)
    print(f'Entropy-> {items}')
    if items.size == 1: #if only 1 row then 0
        return 0
    
    counts = np.zeros((items.shape[0], 1))
    sums = 0
    
    for x in range(items.shape[0]):
        counts[x] = sum(S == items[x]) / (S.size * 1.0)

    for count in counts:
        sums += -1 * count * math.log(count, 2) #calculate entropy 
        
        '''
        Summation(p(xi)* log2(p(xi)))
        '''
        return sums

def gain_ratio(data, col):
    items, dict = subtables(data, col, delete=False) 
                
    total_size = data.shape[0]
    entropies = np.zeros((items.shape[0], 1))
    intrinsic = np.zeros((items.shape[0], 1))
    
    for x in range(items.shape[0]):
        ratio = dict[items[x]].shape[0]/(total_size * 1.0) #ratio 
        '''
        |  |Sv|/|S|  |
        '''
        entropies[x] = ratio * entropy(dict[items[x]][:, -1])  

        '''
        |  |Sv|/|S|  | * entropy(S)
        '''
        intrinsic[x] = ratio * math.log(ratio, 2)

        #calculate entropy 
        
        '''
        Summation(p(xi)* log2(p(xi)))
        '''
        
    total_entropy = entropy(data[:, -1]) #dataset entropy
    iv = -1 * sum(intrinsic) #information gain (i guess... :D)
    
    for x in range(entropies.shape[0]):
        total_entropy -= entropies[x]  #gain 

        
        
        '''
        gain(s) = Totalentropy(S) -I(Attr) 
        '''
        
    return total_entropy / iv

def create_node(data, metadata):
    if (np.unique(data[:, -1])).shape[0] == 1:
        node = Node("")
        node.answer = np.unique(data[:, -1])[0]
        return node

        '''
        if only one row
            return node as leaf
        '''
        
    gains = np.zeros((data.shape[1] - 1, 1))
    
    for col in range(data.shape[1] - 1):
        gains[col] = gain_ratio(data, col)
        '''
        Calculate individual gains or all columns
        
        '''
        
    split = np.argmax(gains) # calculate max gain to make root node
    
    node = Node(metadata[split])  #make root node with max gain  
    metadata = np.delete(metadata, split, 0)    
    
    items, dict = subtables(data, split, delete=True)
    
    for x in range(items.shape[0]):
        child = create_node(dict[items[x]], metadata)
        node.children.append((items[x], child)) 

        '''
        Iteratively call create_node as ID3 is recurrsive algo
        '''
    
    return node

def empty(size):
    s = ""
    for x in range(size):
        s += "   "
    return s

def print_tree(node, level):
    if node.answer != "":
        print(empty(level), node.answer)
        return
    print(empty(level), node.attribute)
    for value, n in node.children:
        print(empty(level + 1), value)
        print_tree(n, level + 2)

metadata, traindata = read_data("tennisdata.csv")
data = np.array(traindata)
node = create_node(data, metadata)
print_tree(node, 0)