import numpy as np


def compute_kappa(gt, predict, class_num=6):
    if not len(gt) == len(predict):
        print("Error: lists not same length: ", gt, predict)
        return 0
    p = [[0]*class_num for _ in range(class_num)]
    for i in range(len(gt)):
        p[gt[i]][predict[i]] += 1
    p = np.array(p)
    print("p", p)
    p0 = sum(np.diagonal(p))/len(gt)
    print(p0)
    pe = 0
    for i in range(class_num):
        pe += sum(p[:, i]) * sum(p[i, :]) #p[:,i],预测为i的个数  p[i,:],真实值为i的个数
    pe = pe / (len(gt) * len(gt))
    k = (p0 - pe)/(1 - pe)
    return k

rater_a = [0, 3, 4, 5, 2, 3, 4, 1, 2, 3, 5, 4, 3, 2, 4, 1, 0, 2, 3, 3]
rater_b = [2, 3, 4, 5, 2, 3, 2, 0, 2, 4, 5, 4, 3, 2, 4, 1, 0, 2, 3, 3]
confusion_matrix = [rater_a, rater_b]
result = compute_kappa(rater_a, rater_b)
print("result:", result)