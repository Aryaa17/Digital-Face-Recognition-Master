


def countTop(sample):
    m,n = sample.shape
    count_zero = 0
    count_X = 0
    count_Plus = 0
    for i in range(int(m/2)):
        for j in range(n):
            if(sample[i][j]==0):
                count_zero = count_zero+1
            if(sample[i][j]==1):
                count_X = count_X+1
            if(sample[i][j]==2):
                count_Plus = count_Plus+1
    return count_zero, count_X, count_Plus



def countBottom(sample):
    m,n = sample.shape
    count_zero = 0
    count_X = 0
    count_Plus = 0
    for i in range(int(m/2)):
        for j in range(n):
            if(sample[int(m/2)+i][j]==0):
                count_zero = count_zero+1
            if(sample[i][j]==1):
                count_X = count_X+1
            if(sample[i][j]==2):
                count_Plus = count_Plus+1
    return count_zero, count_X, count_Plus




def countLeft(sample):
    m,n = sample.shape
    count_zero = 0
    count_X = 0
    count_Plus = 0
    for i in range(m):
        for j in range(int(n/2)):
            if(sample[i][j]==0):
                count_zero = count_zero+1
            if(sample[i][j]==1):
                count_X = count_X+1
            if(sample[i][j]==2):
                count_Plus = count_Plus+1
    return count_zero, count_X, count_Plus



def countRight(sample):
    m,n = sample.shape
    count_zero = 0
    count_X = 0
    count_Plus = 0
    for i in range(m):
        for j in range(int(n/2)):
            if(sample[i][int(n/2)+j]==0):
                count_zero = count_zero+1
            if(sample[i][j]==1):
                count_X = count_X+1
            if(sample[i][j]==2):
                count_Plus = count_Plus+1
    return count_zero, count_X, count_Plus




def reducegreysapce(sample):
    m,n = sample.shape
    begin = np.empty([m], dtype= int)
    ending = np.empty([m], dtype= int)
    Counter1= True
    Counter2 = True
    Counter11= True
    Counter21= True
    column_begin = 0
    column_end = 0
    for i in range(m):
        for j in range(n):
            if (sample[i][j]!=0 and Counter1 == True):
                begin[i]=j
                Counter1 = False
                
            if (sample[i][j]!=0 and Counter11 == True):
                column_begin = i
                Counter11 = False
                
            if (sample[i][n-j-1]!=0 and Counter2 == True):
                 ending[i]=j
                 Counter2 = False
            
            if (sample[n-1-i][j]!=0 and Counter21 == True):
                column_end = n-1-i
                Counter21 = False
                
        Counter1 = True
        Counter2 = True
    initial_counter= np.amin(begin)
    end_counter = np.amax(ending)
    new_sample=[[] for k in range(m)]
    for i in range(m):
        new_sample[i] = sample[i][initial_counter:end_counter]
    new_sample = np.array(new_sample) 
    print(new_sample)
    m1, n1 = new_sample.shape
    l = (column_end-column_begin)/(end_counter-initial_counter)
    return new_sample, l
    
    
def computeFea(sample):
    m,n=sample.shape
    f1=countX(sample)
    f2=countPlus(sample)
    [new_sample, l] = reducegreysapce(sample)
    f3 = countZero(new_sample)
    [a1, b1, c1]= countTop(sample)
    #[a2, b2, c2]= countBottom(sample)
    #[a3, b3, c3]= countLeft(sample)
    [a4, b4, c4]= countRight(sample)
    f4 = (b1+c1)/(a1+b1+c1)
    #f5 = (b2+c2)/(a2+b2+c2)
    #f6 = (b3+c3)/(a3+b3+c3)
    f7 = (b4+c4)/(a4+b4+c4)
    f8 =l
    #xx=[f1, f2, f3, f4, f5]
    xx=[f1, f2, f3, f4, f7, f8]
    #xx=sample.reshape(28*28,1)
    #xx=list(xx)
    return xx
