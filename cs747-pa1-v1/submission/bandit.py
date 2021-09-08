import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ins = "../instances/instances-task1/i-1.txt"
# f2 = "/home/atharva/cs747/cs747-pa1-v1/instances/instances-task1/i-1.txt"
# f3 = "/home/atharva/cs747/cs747-pa1-v1/instances/instances-task1/i-1.txt"

# insta = sys.argv[2]
# al = sys.argv[4]
# rs = int(sys.argv[6])
# ep = float(sys.argv[8])
# c = float(sys.argv[10])
# th = float(sys.argv[12])
# hz = int(sys.argv[14])

def openfile(ins):
    file = open(ins,"r")
    lines = file.readlines()
    inst = []
    for line in lines:
        line = line.rstrip('\n')
        p = float(line)
        inst.append(p)
    file.close()
    instance = np.array(inst)
    return instance

def pull_arm(ins,n):
    p = ins[n]
    op = np.random.choice(np.arange(0,2), p=[1-p,p])
    return op

def epsilong2(ep,hz,ins):
    l = len(ins)
    reward = np.zeros(l)
    num = np.ones(l)
    for k in range(l):
        out = pull_arm(ins,k)
        reward[k]+=out
    t1 = ep*hz
    for t in range(l,hz):
        if(t<=t1):
            n = np.random.choice(np.arange(0,l))
            out = pull_arm(ins,n)
            reward[n]+=out
            num[n]+=1
            avg = reward/num
        else:
            n = avg.argmax()#can add condition for same avg
            out = pull_arm(ins,n)
            reward[n]+=out
            num[n]+=1
            avg = reward/num
    rew = np.sum(reward)
    mcr = hz*np.max(ins)
    reg = mcr-rew
    return reg

def epsilong3(ep,hz,ins):
    l = len(ins)
    reward = np.zeros(l)
    num = np.ones(l)
    for k in range(l):
        out = pull_arm(ins,k)
        reward[k]+=out
    avg = reward/num
    wtd = np.random.choice(np.arange(0,2), p=[ep,1-ep])
    for t in range(l,hz):
        if(wtd==0):
            n = np.random.choice(np.arange(0,l))
            out = pull_arm(ins,n)
            reward[n]+=out
            num[n]+=1
            avg = reward/num
        else:
            n = avg.argmax()#can add condition for same avg
            out = pull_arm(ins,n)
            reward[n]+=out
            num[n]+=1
            avg = reward/num
    rew = np.sum(reward)
    mcr = hz*np.max(ins)
    reg = mcr-rew
    return reg

def ucbf(ins,hz):
    l = len(ins)
    num = np.ones(l)
    reward = np.zeros(l)
    avg = np.zeros(l)
    for k in range(l):
        out = pull_arm(ins,k)
        reward[k]+=out
    avg = reward/num
    ucb=avg

    for t in range(l,hz):
        n = ucb.argmax()
        out = pull_arm(ins,n)
        reward[n]+=out
        num[n]+=1
        avg = reward/num
        ucb = avg + np.sqrt(2*np.log(t)/num)
    rew = np.sum(reward)
    mcr = hz*np.max(ins)
    reg = mcr-rew
    return reg

def getmax(avg,ct,l):
    vals = np.zeros(l)
    def kl(q,j):
        return avg[j]*np.log(avg[j]/q) + (1-avg[j])*np.log((1-avg[j])/(1-q))
    for j in range(l):
        min_val = avg[j]
        max_val = 1
        for i in range(20):
            val = min_val+max_val/2
            if kl(val,j)<=ct[j]:
                min_val = val
            else:
                max_val = val
        vals[j] = val
    return vals

def klucbf(ins,hz):
    l = len(ins)
    num = np.ones(l)
    reward = np.zeros(l)
    avg = np.zeros(l)
    for k in range(l):
        out = pull_arm(ins,k)
        reward[k]+=out
    avg = reward/num
    klucb = avg
    np.seterr(divide = 'ignore')

    for t in range(l,hz):
        n = klucb.argmax()
        out = pull_arm(ins,n)
        reward[n]+=out
        num[n]+=1
        avg = reward/num
        ct = (np.log(t) + 3*np.log(np.log(t)))/num
        klucb = getmax(avg,ct,l)
    rew = np.sum(reward)
    mcr = hz*np.max(ins)
    reg = mcr-rew
    return reg
        
def thompson(ins,hz):
    l = len(ins)
    num = np.ones(l)
    reward = np.zeros(l)
    avg = np.zeros(l)
    for k in range(l):
        out = pull_arm(ins,k)
        reward[k]+=out
    avg = reward/num
    succ = np.ones(l)
    fail = np.ones(l)
    pval = np.copy(avg)
    for t in range(hz):
        n = np.argmax(pval)
        out = pull_arm(ins,n)
        reward[n]+=out
        num[n]+=1
        if(out==1):
            succ[n]+=1
        else:
            fail[n]+=1
        pval = np.random.beta(succ,fail)
    rew = np.sum(reward)
    mcr = hz*np.max(ins)
    reg = mcr-rew
    return reg
        
if __name__ == "__main__":
    # ins = openfile(insta)
    # if(al=="epsilon-greedy-t1"):
    #     reg = epsilong3(ep,hz,ins)
    #     print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    # if(al=="ucb-t1"):
    #     reg = ucbf(ins,hz)
    #     print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    # if(al=="kl-ucb-t1"):
    #     reg = klucbf(ins,hz)
    #     print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    # if(al=="thompson-sampling-t1"):
    #     reg = thompson(ins,hz)
    #     print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    
    instances = ["../instances/instances-task1/i-1.txt", "../instances/instances-task1/i-2.txt", "../instances/instances-task1/i-3.txt"]
    algorithms = ["epsilon-greedy-t1", "ucb-t1", "kl-ucb-t1", "thompson-sampling-t1"]
    horizons = [100,400,1600,6400,25600,102400]
    ep = 0.02
    filew = open("output_t1.txt","a")
    i = 1
    for insta in instances:
        j=0
        ins = openfile(insta)
        plt.figure()
        plt.xlabel("Horizon")
        plt.ylabel("Regret")
        name = ["epsilon-greedy","ucb","kl-ucb","thompson-sampling"]
        for al in algorithms:
            mregrs = []
            for hz in horizons:
                regrs = []
                for rs in range(50):
                    np.random.seed(rs)
                    if(al=="epsilon-greedy-t1"):
                        reg = epsilong3(ep,hz,ins)
                        filew.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(insta, al, rs, ep, 2, 0, hz, reg, 0))
                    if(al=="ucb-t1"):
                        reg = ucbf(ins,hz)
                        filew.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(insta, al, rs, ep, 2, 0, hz, reg, 0))
                    if(al=="kl-ucb-t1"):
                        reg = klucbf(ins,hz)
                        filew.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(insta, al, rs, ep, 2, 0, hz, reg, 0))
                    if(al=="thompson-sampling-t1"):
                        reg = thompson(ins,hz)
                        filew.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(insta, al, rs, ep, 2, 0, hz, reg, 0))
                    regrs.append(reg)
                regrs = np.array(regrs)
                mreg = np.mean(regrs)
                mregrs.append(mreg)
            plt.scatter(horizons, mregrs)
            plt.plot(horizons, mregrs, label=name[j])
            j+=1
        plt.yscale("log")
        plt.legend(loc="upper left")
        plt.savefig('ins{}'.format(i))
        i+=1
        plt.show()
            
            


    




    # if(al=="ucb-t2"):
    #     reg = epsilong3(ep,hz,ins)
    #     print(reg)
    # if(al=="alg-t3"):
    #     reg = epsilong3(ep,hz,ins)
    #     print(reg)
    # if(al=="alg-t4"):
    #     reg = epsilong3(ep,hz,ins)
    #     print(reg)