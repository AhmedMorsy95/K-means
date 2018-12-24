import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
from mpl_toolkits.mplot3d import Axes3D
import math




class Kmeans:
    def __init__(self,k,max_iterations,data):
        self.k = k
        self.max_iterations = max_iterations
        self.centers = []
        # random k centers
        for i in range(0,k):
            self.centers.append(data[i])
        self.data = data
        self.n = len(data)
        self.d = len(data[0])
        self.nearest_center = [0]*self.n


    def update_centers(self):
        for k in range(0,self.k): # find averge for each center
            kam_point = 0
            sum_of_points = [0]*self.d
            for i in range(0,self.n): # loop over all points
                if self.nearest_center[i] == k:
                    kam_point+=1
                    for j in range(0,self.d):
                        sum_of_points[j] += self.data[i][j]
            if kam_point > 0:
                for j in range(0,self.d):
                    sum_of_points[j]/=kam_point
                self.centers[k] = sum_of_points
        return self.centers

    def update_labels(self):  # matches each point to closest center
        for i in range(0,self.n): # pass over each element
            mn_distance = 1e18
            best = 0
            for k in range(0,self.k): # pass over all centers
                cur = distance(self.data[i],self.centers[k])
                if cur < mn_distance: # if this center is closer update my label
                    mn_distance = cur
                    self.nearest_center[i] = k


    def get_distortion(self):
        # find mean for each cluster
        distortion = 0
        for k in range(0,self.k):
            kam_point = 0
            sum_of_points = [0]*self.d
            for i in range(0,self.n): # loop over all points get points belong to this cluster , sum them
                if self.nearest_center[i] == k:
                    kam_point+=1
                    for j in range(0,self.d):
                        sum_of_points[j] += self.data[i][j]
            for j in range(0,self.d): # get average
                sum_of_points[j]/=kam_point
            if kam_point > 0:
                for i in range(0,self.n): # get distance
                    if self.nearest_center[i] == k:
                            distortion +=  distance(sum_of_points, self.data[i]) / kam_point
        return distortion

    def run_kmeans(self):  # runs and plots distortion
        x = []
        for i in range(0,self.max_iterations):
            self.update_labels()
            self.centers = self.update_centers()
            x.append(self.get_distortion())
            print(self.get_distortion())
        self.plot(x)

    def plot(self,x):
        #plotting distorion function
        plt.xlabel("iterations")
        plt.ylabel("distorion")
        plt.plot(x)
        plt.show()


def distance(a,b):
    ret = 0
    squared_distance = 0
    for i in range(len(a)):
        squared_distance += (a[i] - b[i])**2
    ret = squared_distance
    return ret

def read_data():
    with (open("data_batch_1", "rb")) as openfile:
        while True:
            try:
                dict = pickle.load(openfile,encoding='bytes')
            except EOFError:
                break
    return dict


#our main function
def main():
    K = 10 # clusters
    dict = read_data()
    data = dict.get(b'data') # data
    kmeans = Kmeans(K,20,data)
    kmeans.run_kmeans()

if __name__== "__main__":
  main()
