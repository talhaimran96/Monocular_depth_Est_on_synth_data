import os
import pandas as pd

dataset_path="../../../Desktop/Renders/"

file_list=[]
depth_map_list=[]


for dir in os.listdir(dataset_path):
    for scene in os.listdir(dataset_path+dir):
        for file in os.listdir(f"{dataset_path}/{dir}/{scene}/Images/"):
            # print(f"{dataset_path}/{dir}/{scene}/Images/{file}")
            file_list.append(f"{dataset_path}/{dir}/{scene}/Images/{file}")
            depth_map_list.append(f"{dataset_path}/{dir}/{scene}/Depth_maps/{file}")

    df=pd.DataFrame({"Images":file_list,"Depth_maps":depth_map_list})
    df.to_csv(f"{dataset_path}/{dir}.csv")
    file_list = []
    depth_map_list = []

