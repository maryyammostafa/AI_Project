import numpy as np;

#create a single point crossover function 
def singlePointCrossover(parent1,parent2,rng):
    if len(parent1)!=len(parent2):
        raise ValueError("Parents must have the same length")
    
    point=rng.integers(1,len(parent1))
    child1=np.concatenate((parent1[:point],parent2[point:]))
    child2=np.concatenate((parent2[:point],parent1[point:]))
    return child1, child2



#create a double point crossover function 
def doublePointCrossover(parent1,parent2,rng):
    if len(parent1)!=len(parent2):
        raise ValueError("Parents must have the same length")
    if len(parent1)<3:
        raise ValueError("Parents must have at least 3 genes(features) for double point crossover")
    
    
    points=rng.choice(range(1,len(parent1)),size=2,replace=False) 
    point1,point2=np.sort(points)
    child1=np.concatenate((parent1[:point1],parent2[point1:point2],parent1[point2:]))
    child2=np.concatenate((parent2[:point1],parent1[point1:point2],parent2[point2:]))
    return child1, child2


#create a uniform crossover function 
def uniformCrossover(parent1,parent2,rng):
    if len(parent1)!=len(parent2):
        raise ValueError("Parents must have the same length")
    
    
    random_mask=rng.integers(0,2,size=len(parent1)).astype(bool)
    
    
    child1=np.where(random_mask,parent1,parent2)
    child2=np.where(random_mask,parent2,parent1)
    return child1, child2





