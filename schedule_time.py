import pandas as pd
import numpy as np
import time
import tkinter.constants, tkinter.filedialog
import sys
import random

def schedule_time(pt, m_sequence, j_sequence):
    # s{machine:job 소요 시간}
    # D{(job: machine): job이 machine끝나는 시간}
    # v : job에 걸린 총 시간
    s, D = {}, {}
    v = 0

    for j in range(len(j_sequence)):
        for i in range(len(m_sequence)):
            D[i,j] = 0

    # 첫번째 job이 끝나는데 걸리는 시간을 계산
    for i in range(len(m_sequence)):
        # 첫번째 job의 각 현재 machine에서 작업 시간을 update
        s[i] = pt[j_sequence[0]][m_sequence[i]]
        # 이전 machine에서 끝난 시간 + 현재 machine에서 끝나는데 걸린 시간을 update
        v = v + s[i]
        # machine에서 끝난 시간을 update
        D[i, 0] = v


    # 첫번째 machine에서 각 job을 끝내는데 걸리는 시간 계산
    for j in range(0, len(j_sequence)-1):
        # j번째 job을 끝낸시간에 j+1번쨰 job 시간을 더하여 다음 job이 끝나는 시간 계산

        D[0, j+1] = D[0, j] + pt[j+1][0]

    # 첫번째 job에 대한 schedule은 이전에 저장했으므로 두번째 job 부터 시작
    for j in range(1, len(j_sequence)):
        # j번째 job의 각 machine에서 scheduling
        for i in range(0, len(m_sequence)-1):
            # j번쨰 job의 i번째 machine에서의 작업 시간을 업데이트
            # i번째 까지는 j번째 작업의 시간을 업데이트 하고 i+1부터는 j-1번째 작업의 시간이 저장된 상태
            s[i] = pt[j_sequence[j]][m_sequence[i+1]]
            # j-1번째 작업이 끝난 시간과 i-1번째 machine이 j번째 작업을 끝낸 시간에 현재 job의 시간을 더하여 큰 값을 저장
            D[i+1, j] = max(D[i+1, j-1] + s[i], D[i, j] + s[i])

    v = D[len(m_sequence)-1, len(j_sequence)-1]

    return v

#list all sequence after every motion
# j_sequence : job sequence, V : [0, 0, 0, 0, 0, 0, 0, 0, 0], Q : [0, 0, 0, 0, 0, 0, 0, 0, 0] x 9, epsilon : epsilon
def permutation(j, V, Q, epsilon):
    # job sequence
    vec = j[:]
    # Q 배열 저장
    Q_temp = [[Q[x][y] for y in range(len(Q[0]))] for x in range(len(Q))]

    for i in j:
        # larger then epsilon repersent choosing by best way
        if random.uniform(0, 1) > epsilon:
            # first j is [0, 1, 2, 3, 4, 5, 6, 7, 8]
            # if it is the first choice, according to V
            if j.index(i) == 0:
                # temp : V에서 최소값의 index를 temp에 저장
                temp = V.index(min(float(s) for s in V))
                for k in range(0,len(Q_temp[0])):
                    Q_temp[k][temp] = sys.maxsize

                a, b = 0, vec.index(temp)
                vec[b], vec[a] = vec[a], vec[b]
            # else according to Q
            else:
                temp = Q_temp[i - 1].index(lowest_cost(Q_temp,i - 1))
                for k in range(0,len(Q_temp[0])):
                    Q_temp[k][temp] = sys.maxsize
                a, b = i, vec.index(temp)
                vec[b], vec[a] = vec[a], vec[b]
        # else by random
        else:
            # vec에서 랜덤으로 값을 뽑아서 temp에 저장
            temp = random.choice(vec)
            # 현재 i값과 vec[temp]값을 a, b에 저장
            a, b = i, vec.index(temp)
            # vec의 a, b 인덱스를 바꿔줌
            vec[b], vec[a] = vec[a], vec[b]
    return vec

#get best cost in series sequence
# Q : [0, 0, 0, 0, 0, 0, 0, 0, 0] x 9, r : 특정 job
def lowest_cost(Q, r):
    # sys.maxsize : 9223372036854775807
    cost = sys.maxsize
    for i in range(0, len(Q[0])):
        if Q[r][i] < cost:
            cost = Q[r][i]
    return cost

# update Q
# Q : [0, 0, 0, 0, 0, 0, 0, 0, 0] x 9, r : 섞인 job 순서, cost : total cost time(249)
def update_Q(Q, r, cost):
    gamma = 0.8
    alpha = 0.1
    for i in range(0, len(r) - 1):
        # Q[r[i]][r[i+1]](0) = Q[r[i]][r[i+1]](0) + 0.1 * (cost(249) + 0.8 * lowest_cost()(0) - Q[r[i]][r[i+1]](0))
        Q[r[i]][r[i+1]] = Q[r[i]][r[i+1]] + alpha * (cost + gamma * lowest_cost(Q,r[i]) - Q[r[i]][r[i+1]])
    return Q

# update V
# V : [0, 0, 0, 0, 0, 0, 0, 0, 0], Q : [0, 0, 0, 0, 0, 0, 0, 0, 0] x 9, r : 섞인 job 순서, cost : total cost time(249)
def update_V(V, Q, r, cost):
    gamma = 0.8
    alpha = 0.1
    # V[r[0]](0) = V[r[0]](0) + 0.1 * (cost(249) + 0.8 * lowest_cost()(0) - V[r[0]](0))
    V[r[0]] = V[r[0]] + alpha * (cost + gamma * lowest_cost(Q,r[0]) - V[r[0]])
    return V

filename ='9X5_flowshop.xlsx'#tkFileDialog.askopenfilename(initialdir = "/",title = "Select file")
pt_tmp=pd.read_excel(filename,sheet_name="S1",index_col =[0])
pt = pt_tmp.as_matrix().tolist()

m_sequence = list(range(0,len(pt[0]))) # m_sequence represent machine order in each job
j_sequence = list(range(0,len(pt))) #j_sequence repersent job order
start_sequence = list(range(0,len(pt)))

time_cost = schedule_time(pt, m_sequence, j_sequence)

s = [] #define final state series
print (time_cost)

#start main function
start_time = time.time()
final_cost = sys.maxsize
final_s = []
final_epoch = 0

for epo in range(0,50):
    epsilon = 1 # random threshold, initial by 1
    #QL main function

    Q = []
    V = []
    for i in range(0,len(pt)):
        Q_temp = []
        V.append(0)
        for j in range(0,len(pt)):
            Q_temp.append(0)
        Q.append(Q_temp)


    for loop_times in range(0,10000):
        epsilon = epsilon*0.999 # after one loop, threshold desend by *0.999999
        # j_sequence : job sequence, V : [0, 0, 0, 0, 0, 0, 0, 0, 0], Q : [0, 0, 0, 0, 0, 0, 0, 0, 0] x 9, epsilon : epsilon
        # r : 섞인 job 순서
        r = permutation(j_sequence, V, Q, epsilon)
        # pt : data table, m_sequence, r: permutation에서 섞인 job 순서
        # cost : total cost time
        cost = schedule_time(pt, m_sequence, r)
        # V : [0, 0, 0, 0, 0, 0, 0, 0, 0], Q : [0, 0, 0, 0, 0, 0, 0, 0, 0] x 9, r : 섞인 job 순서, cost : total cost time(249)
        V = update_V(V, Q, r, cost)
        # Q : [0, 0, 0, 0, 0, 0, 0, 0, 0] x 9, r : 섞인 job 순서, cost : total cost time(249)
        Q = update_Q(Q, r, cost)

    s_result = []
    choose_index = V.index(min(float(s) for s in V))
    s_result.append(choose_index)
    for i in range(0,len(V)):
        Q[i][choose_index] = sys.maxsize
    for i in range(1,len(V)):
        next_index = Q[choose_index].index(min(float(s) for s in Q[choose_index]))
        s_result.append(next_index)
        for j in range(0,len(V)):
            Q[j][next_index] = sys.maxsize
        choose_index = next_index
    now_cost = schedule_time(pt, m_sequence, s_result)
    if final_cost > now_cost:
        final_cost = now_cost
        final_s = s_result
        final_epoch = epo
    print ("The %d epoch result is %s , cost is %d" % (epo, s_result, now_cost))
end_time = time.time()
print ("Ending learning. The best result by learning is %s at number %d epoch, cost is %d" % (final_s, final_epoch, final_cost))
print ("Total time cost: %f s" % (end_time - start_time))

