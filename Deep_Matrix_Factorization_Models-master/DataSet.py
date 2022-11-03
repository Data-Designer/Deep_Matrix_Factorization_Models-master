# -*- Encoding:UTF-8 -*-

import numpy as np
import sys

from scipy import io
import os



class DataSet(object):
    def __init__(self, fileName):
        self.data, self.shape = self.getData(fileName)
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getTrainDict()

    # def getData(self, fileName):
    #     if fileName == 'ml-1m':
    #         print("Loading ml-1m data set...")
    #         data = []
    #         filePath = './Data/ml-1m/ratings.dat'
    #         u = 0
    #         i = 0
    #         maxr = 0.0
    #         with open(filePath, 'r') as f:
    #             for line in f:
    #                 if line:
    #                     lines = line[:-1].split("::")
    #                     user = int(lines[0])
    #                     movie = int(lines[1])
    #                     score = float(lines[2])
    #                     time = int(lines[3])
    #                     data.append((user, movie, score, time))
    #                     if user > u:
    #                         u = user
    #                     if movie > i:
    #                         i = movie
    #                     if score > maxr:
    #                         maxr = score
    #         self.maxRate = maxr
    #         print("Loading Success!\n"
    #               "Data Info:\n"
    #               "\tUser Num: {}\n"
    #               "\tItem Num: {}\n"
    #               "\tData Size: {}".format(u, i, len(data)))
    #         return data, [u, i]
    #     else:
    #         print("Current data set is not support!")
    #         sys.exit()

    # def getData(self,path):
    #     # Groupbuying
    #     data = io.loadmat(path)  # 读取交互数据
    #     data_active = io.loadmat('Data/ml-1m/Groupbuying/active_user.mat')  # 读取active_id
    #     data_act = data_active['active_user'].astype(int)
    #     # 下面需要实现一个映射表
    #     data_map = {}  # 旧id-新id
    #     for i in range(len(data_act)):
    #         data_map[data_act[i][1]] = data_act[i][0]
    #     user_to_item = {}  # 交互列表
    #     # 以dict形式保存
    #     for index, i in enumerate(data['U_I'], start=1):
    #         user_to_item[index] = i[0][0].tolist()
    #     # 下面是为了与源代码返回保持一致
    #     user_num = 3912
    #     item_num = 1040
    #     data = []  # 重新分配data内存[(user,item,time)]
    #     for userid in range(1, len(user_to_item)):
    #         if userid not in data_map.keys():
    #             continue
    #         flag = 1  # 没有时间戳，所以只能以0-1-2代替
    #         for itemid in user_to_item[userid]:
    #             data.append((data_map[userid], itemid, 1, flag))  # usr,item,1,time；从1开始标号
    #             flag = flag + 1
    #     self.maxRate = 1
    #     return data, [user_num, item_num]
    def getData(self,path):
        data = io.loadmat(path)  # 读取交互数据
        data_active = io.loadmat('Data/ml-1m/P2P1700/active_user.mat')  # 读取active_id
        data_act = data_active['active_user'].astype(int)[:, 0]
        user_map = {}  # 必须重索引
        for index, user_id in enumerate(data_act, start=1):
            user_map[user_id] = index
        data_available_item = io.loadmat('Data/ml-1m/P2P1700/available_item.mat')  # 读取available_item_id
        data_available = data_available_item['avaliable_item'].astype(int)[:, 0]
        item_map = {}  # 同上
        for index, item_id in enumerate(data_available, start=1):
            item_map[item_id] = index
        user_to_item = {}  # 交互列表
        # 以dict形式保存
        for index, i in enumerate(data['U_I'], start=1):
            user_to_item[index] = i[0][0].tolist()  # {1:[itemid,item_id]}
        # 下面是为了与源代码返回保持一致
        user_num = 3607 # 3607
        item_num = 544 # 544
        data = []  # 重新分配data内存[(user,item,score,time)]
        for userid in range(1, len(user_to_item)):
            if userid not in data_act:  # 如果不在
                continue
            flag = 1  # 没有时间戳，所以只能以0-1-2代替
            for itemid in user_to_item[userid]:
                if itemid in data_available:  # 只保存活跃的item
                    data.append((user_map[userid], item_map[itemid], 1, flag))  # usr,item,1,time；从1开始标号
                flag = flag + 1
        self.maxRate = 1
        return data, [user_num, item_num]  # user从0开始的

    def getTrainTest(self):
        data = self.data # user, movie, score, time
        data = sorted(data, key=lambda x: (x[0], x[3]))
        train = []
        test = []
        for i in range(len(data)-1):
            user = data[i][0]-1 # 这里选择都从0开始index
            item = data[i][1]-1
            rate = data[i][2]
            if data[i][0] != data[i+1][0]:
                test.append((user, item, rate)) # 每个用户留了一个项目用作测试，这个交互过的项目应该是最高的才对。
            else:
                train.append((user, item, rate))

        test.append((data[-1][0]-1, data[-1][1]-1, data[-1][2]))
        return train, test

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getEmbedding(self):
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating # 训的时候是一个完整的矩阵，感觉是作为训练的label使用的一个查找表格
        return np.array(train_matrix)

    def getInstances(self, data, negNum):
        user = []
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate) # 返回负采样的item,这里是用于训练的负样本

    def getTestNeg(self, testData, negNum):
        user = []
        item = []
        for s in testData: # s:(user, item, rate) 正例
            tmp_user = []
            tmp_item = []
            u = s[0]
            i = s[1]
            tmp_user.append(u) # [user_id]
            tmp_item.append(i) #[posId]
            neglist = set()
            neglist.add(i) # [posId]
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (u, j) in self.trainDict or j in neglist:
                    j = np.random.randint(self.shape[1]) # 保证负例不重复
                neglist.add(j)
                tmp_user.append(u) # [userId,userId,] 100个
                tmp_item.append(j) # # [posId,negId,...] # 100个
            user.append(tmp_user) # [[userID,...],[usrID2,...]]
            item.append(tmp_item) # [[posId,negId,...],[posId,negId,...]]
        return [np.array(user), np.array(item)] # 这里是用于Test评测指标计算时候使用的neg item+ 正例
