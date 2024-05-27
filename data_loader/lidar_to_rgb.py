import os
import random
import numpy as np
import torch.utils.data as data
import csv
import cv2 as cv
import matplotlib.pyplot as plt


from data_loader.loader_utils import *

__all__ = ['RELLIS_3D']


class RELLIS_3D(data.Dataset):
    """
    Args:
        mode:
        process_data (callable):
        generate_data (callable):
        args:
    """
    def __init__(self, mode, args):
        self.mode = mode
        self.process_data = ProcessRELLIS(args)
        self.data_path = '/Users/baekchaeyoon/Desktop/Univ/VIP_Lab/RELLIS_data/data/RELLIS-3D'
        # self.data_path = args["data_root"]
        
        if mode == "train":
            self.num_samples = args['train_samples']
            self.delta_ij_max = args['delta_ij_max']
            self.translation_max = args['translation_max']
            self.accumulation_frame_num = args['accumulation_frame_num']
            self.accumulation_frame_skip = args['accumulation_frame_skip']
            self.samples = self.make_sample_dataset()
        elif mode == 'valid':
            self.num_samples = args['val_samples']
            self.delta_ij_max = args['delta_ij_max']
            self.translation_max = args['translation_max']
            self.accumulation_frame_num = args['accumulation_frame_num']
            self.accumulation_frame_skip = args['accumulation_frame_skip']
            self.samples = self.make_sample_dataset()
        elif mode == 'test':
            self.num_samples = args['val_samples']
            self.accumulation_frame_num = args['accumulation_frame_num']
            self.accumulation_frame_skip = args['accumulation_frame_skip']       
            self.rand_init = {}
            f = open(args["rand_init"], 'r')
            rdr = csv.reader(f)
            for k, line in enumerate(rdr):
                self.rand_init[line[0]] = [float(i) for i in line[1:]] 
            f.close()        
            self.samples = self.make_test_sample_dataset(self.rand_init)
        else:
            print('worng mode: ', mode)
            exit()
        
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_path + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pcd, img, calib_seq, posej_T_posei, fname = self.file_reader(self.samples[index]) 
        if self.mode != 'test': rand_init = None                 
        else: rand_init = self.rand_init[fname]  
        return self.process_data(pcd, img, calib_seq, posej_T_posei, fname, rand_init=rand_init)

    def get_sequence_j(self, poses, seq_i):
        seq_sample_num = len(poses)
        # get the max and min of possible j
        seq_j_min = max(seq_i - self.delta_ij_max, 0)
        seq_j_max = min(seq_i + self.delta_ij_max, seq_sample_num - 1)

        # pose of i
        Pi = pose_read(poses[seq_i])

        while True:
            seq_j = random.randint(seq_j_min, seq_j_max)
            # get the pose, if the pose is too large, ignore and re-sample
            Pj = pose_read(poses[seq_j])
            posej_T_posei = np.linalg.inv(Pj) @ Pi # 4x4
            t_ji = posej_T_posei[0:3, 3]  # 3
            t_ji_norm = np.linalg.norm(t_ji)  # scalar
            if t_ji_norm < self.translation_max: break
            else: continue

        return seq_j, posej_T_posei

    def make_test_sample_dataset(self, rand_init):       

        cam_name_dict = {}
        for seq in [0,1,2,3,4]:      
            if (int(seq) in cam_name_dict.keys()) == False: cam_name_dict[int(seq)] = {}
            file_list = os.listdir(os.path.join(self.data_path, "Rellis-3D", str(seq).zfill(5), "pylon_camera_node"))
            for file_one in file_list:
                fn = file_one.split("/")[-1]
                cam_name_dict[int(seq)][fn[5:11]] = fn[:-4]   

        K_scale= np.eye(4)
        K_scale[0,0] = 1600./ 1920.
        K_scale[1,1] = 900. / 1200.

        calib_dict = {}
        for seq in [0,1,2,3,4]:       
            calib_dict[seq] = {}  

            Tr_fn = os.path.join(self.data_path, "Rellis_3D", str(seq).zfill(5), "transforms.yaml")
            RT = get_lidar2cam_mtx(Tr_fn)
            calib_dict[seq]["Tr"] = RT
            calib_dict[seq]["Tr_inv"] = np.linalg.inv(RT)

            P_fn = os.path.join(self.data_path, "Rellis-3D", str(seq).zfill(5), "camera_info.txt")
            P = get_cam_mtx(P_fn)
            P_eye = np.eye(4)
            P_eye[:3,:3] = P
            P_eye = K_scale @ P_eye
            calib_dict[seq]["P"] = P_eye
            calib_dict[seq]["P_inv"] = np.linalg.inv(P_eye)     

        print("출력용 ----------------------------------------------------------")
        for i in range(5): 
            print(f"시퀀스 {i}:")
            print(f"Tr (라이다에서 카메라로의 변환 매트릭스):\n{calib_dict[i]['Tr']}")
            print(f"Tr_inv (라이다에서 카메라로의 변환 매트릭스의 역행렬):\n{calib_dict[i]['Tr_inv']}")
            print(f"P (카메라 내부 매개변수 매트릭스):\n{calib_dict[i]['P']}")
            print(f"P_inv (카메라 내부 매개변수 매트릭스의 역행렬):\n{calib_dict[i]['P_inv']}")
            print(" ")  
            
        sample_list = []
        for seq in [0,1,2,3,4]:           
            seq_str = str(seq).zfill(5)   
            f = open(os.path.join(self.data_path, "Rellis-3D", seq_str, "poses.txt"), 'r')
            poses = f.readlines()
            f.close()
            for k in rand_init.keys():
                seq_key, seq_i, seq_j = int(k.split("_")[0]), int(k.split("_")[1]), int(k.split("_")[2])
                if seq_key != seq : continue 

                Pi = pose_read(poses[seq_i])
                Pj = pose_read(poses[seq_j])
                posej_T_posei = np.linalg.inv(Pj) @ Pi # 4x4

                str_seq_i = str(seq_i).zfill(6)
                str_seq_j = str(seq_j).zfill(6)  

                indiv_sample = {'image': os.path.join(self.data_path, "Rellis-3D", seq_str, "pylon_camera_node", cam_name_dict[seq][str_seq_j] + '.jpg'),
                                "os1": os.path.join(self.data_path, "Rellis-3D", seq_str, "os1_cloud_node_kitti_bin", str_seq_i + '.bin'),
                                'calib': calib_dict[seq],
                                'posej_T_posei': posej_T_posei,
                                'fname': seq_str + '_' + str_seq_i + '_' + str_seq_j}
                sample_list.append(indiv_sample)
        sample_list = sample_list[:self.num_samples]
        return sample_list

    def make_sample_dataset(self):  

        # 로그 확인 용
        print("Creating sample dataset...")
        
        ptname = "pt_" + self.mode + ".lst"
        if self.mode == "valid": ptname = "pt_val.lst"
        f = open(os.path.join(self.data_path, ptname), 'r')
        split_list = f.readlines()
        f.close()

        sample_split_dict = {}
        for split_ in split_list:
            split_os1_fn = split_.split(" ")[0]
            seq, _, fn = split_os1_fn.split("/")
            if (int(seq) in sample_split_dict.keys()) == False: sample_split_dict[int(seq)] = []
            sample_split_dict[int(seq)].append(int(fn[:-4]))

        cam_name_dict = {}
        for seq in sample_split_dict.keys():      
            if (int(seq) in cam_name_dict.keys()) == False: cam_name_dict[int(seq)] = {}
            file_list = os.listdir(os.path.join(self.data_path, "Rellis-3D", str(seq).zfill(5), "pylon_camera_node"))
            for file_one in file_list:
                fn = file_one.split("/")[-1]
                cam_name_dict[int(seq)][fn[5:11]] = fn[:-4]       

        K_scale= np.eye(4)
        K_scale[0,0] = 1600./ 1920.
        K_scale[1,1] = 900. / 1200.

        calib_dict = {}
        for seq in sample_split_dict.keys():    
            calib_dict[seq] = {}  

            Tr_fn = os.path.join(self.data_path, "Rellis_3D", str(seq).zfill(5), "transforms.yaml")
            RT = get_lidar2cam_mtx(Tr_fn)
            calib_dict[seq]["Tr"] = RT
            calib_dict[seq]["Tr_inv"] = np.linalg.inv(RT)

            P_fn = os.path.join(self.data_path, "Rellis-3D", str(seq).zfill(5), "camera_info.txt")
            P = get_cam_mtx(P_fn)
            P_eye = np.eye(4)
            P_eye[:3,:3] = P
            P_eye = K_scale @ P_eye
            calib_dict[seq]["P"] = P_eye
            calib_dict[seq]["P_inv"] = np.linalg.inv(P_eye)        

        sample_list = []
        for seq in sample_split_dict.keys():         
            seq_str = str(seq).zfill(5)   
            f = open(os.path.join(self.data_path, "Rellis-3D", seq_str, "poses.txt"), 'r')
            poses = f.readlines()
            f.close()

            file_list = sample_split_dict[seq]
            for seq_i in file_list:

                seq_j, posej_T_posei = self.get_sequence_j(poses, seq_i)

                str_seq_i = str(seq_i).zfill(6)
                str_seq_j = str(seq_j).zfill(6)  

                indiv_sample = {'image': os.path.join(self.data_path, "Rellis-3D", seq_str, "pylon_camera_node", cam_name_dict[seq][str_seq_j] + '.jpg'),
                                "os1": os.path.join(self.data_path, "Rellis-3D", seq_str, "os1_cloud_node_kitti_bin", str_seq_i + '.bin'),
                                'calib': calib_dict[seq],
                                'posej_T_posei': posej_T_posei,
                                'fname': seq_str + '_' + str_seq_i + '_' + str_seq_j}
                sample_list.append(indiv_sample)
        if self.mode == "train" or self.mode == "valid" : 
            random.shuffle(sample_list)      

        if self.num_samples > 0:
            sample_list = sample_list[:self.num_samples]
        else:
            self.num_samples = len(sample_list)
        return sample_list

    def search_for_accumulation(self, pcd_dir, poses, seq_i, seq_sample_num, P_oi, stride):

        P_io = np.linalg.inv(P_oi)

        pc_np_list = []

        counter = 0
        while len(pc_np_list) < self.accumulation_frame_num:
            counter += 1
            seq_j = seq_i + stride * counter
            if seq_j < 0 or seq_j >= seq_sample_num:
                break

            str_seq_j = str(seq_j).zfill(6) 
            # print('str_seq_ij: ', seq_i, str_seq_j)

            pc_j = pcd_read(os.path.join(pcd_dir, str_seq_j + '.bin'))
            pc_j = pc_j.T
            P_oj = pose_read(poses[seq_j])
            P_ij = P_io @ P_oj

            pc_j = np.concatenate((pc_j[:3, :], np.ones((1, pc_j.shape[1]), dtype=pc_j.dtype)), axis=0)
            pc_j = P_ij @ pc_j
            pc_j = pc_j[:3, :]

            pc_np_list.append(pc_j)

        return pc_np_list

    def get_accumulated_pc(self, pcd_path, seq, seq_i):

        pc_np = pcd_read(pcd_path)
        pc_np = pc_np.T
        # shuffle the point cloud data, this is necessary!
        pc_np = pc_np[:, np.random.permutation(pc_np.shape[1])]
        pc_np = pc_np[:3, :]  # 3xN

        if self.accumulation_frame_num <= 0.5: return pc_np.T

        pc_np_list = [pc_np]

        # pose of i
        f = open(os.path.join(self.data_path, "Rellis-3D", str(seq).zfill(5) , "poses.txt"), 'r')
        poses = f.readlines()
        f.close()
        seq_sample_num = len(poses)

        P_oi = pose_read(poses[seq_i])   

        pcd_paths = pcd_path.split('/')[:-1]
        pcd_dir = os.path.join('/', *pcd_paths)
        # search for previous
        prev_pc_np_list = self.search_for_accumulation(pcd_dir, poses, seq_i, seq_sample_num,
                                                       P_oi, -self.accumulation_frame_skip)
        # search for next
        next_pc_np_list = self.search_for_accumulation(pcd_dir, poses, seq_i, seq_sample_num,
                                                       P_oi, self.accumulation_frame_skip)

        pc_np_list = pc_np_list + prev_pc_np_list + next_pc_np_list

        pc_np = np.concatenate(pc_np_list, axis=1)

        return pc_np.T

    def file_reader(self, sample_one):
        """
        :param sample_path:
        :return:
        """
        seq, str_seq_i = sample_one['fname'].split('_')[0], sample_one['fname'].split('_')[1]
        pcd = self.get_accumulated_pc(sample_one['os1'], seq, int(str_seq_i))
        img = rgb_read(sample_one['image'])
        return pcd, img, sample_one['calib'], sample_one['posej_T_posei'], sample_one['fname']

class ProcessRELLIS(object):

    def __init__(self, args):
        self.raw_cam_img_size = args['raw_cam_img_size']
        self.lidar_line = args['lidar_line']
        self.num_points = args['num_points']
        if args['test'] == False:
            self.l_rot_range = args['dclb']['l_rot_range']
            self.l_trs_range = args['dclb']['l_trs_range']
            self.c_rot_range = args['dclb']['c_rot_range']
        else:
            self.l_rot_range, self.l_trs_range, self.c_rot_range = None, None, None
        return     

    def __call__(self, pcd, img, calib_seq, posej_T_posei, fname, rand_init=None):

        # 변환 매트릭스 추출
        Tr = calib_seq['Tr']
        P = calib_seq['P']
        
        # LiDAR 포인트를 카메라 좌표계로 변환합니다.
        ones = np.ones((pcd.shape[0], 1))
        pcd_homogeneous = np.hstack((pcd[:, :3], ones))  # 포인트에 동차 좌표 추가
        pcd_camera = Tr.dot(pcd_homogeneous.T)

        # 카메라 내부 매개변수를 사용하여 2D 이미지 평면에 투영합니다.
        pcd_projected = P.dot(pcd_camera)
        pcd_projected /= pcd_projected[2, :]  # Z 값으로 나누어 정규화

        # 2D 이미지 좌표 추출
        x = pcd_projected[0, :].astype(int)
        y = pcd_projected[1, :].astype(int)

        # 이미지에 포인트 그리기
        for xi, yi in zip(x, y):
            if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
                cv.circle(img, (xi, yi), 3, (255, 0, 0), -1)  # 빨간색 점으로 표시

        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title(f"Visualized Image: {fname}")
        plt.axis('off')
        plt.show()

        return img
    
def get_default_args():
    return {
        'data_root': "/Users/baekchaeyoon/Desktop/Univ/VIP_Lab/RELLIS_data/data",
        'num_points': 65536,
        'lidar_line': 64,
        'lidar_fov_rad': [0.125, -0.125],
        'raw_cam_img_size': [900, 1600],
        'val_samples': -1,  
        'test' : mode == 'test',
        'delta_ij_max': 40,
        'translation_max': 10,
        'accumulation_frame_num': 0,
        'accumulation_frame_skip': 1,
        'rand_init': '/Users/baekchaeyoon/Desktop/Univ/VIP_Lab/RELLIS_data/EFGH/params/rellis3d_rand_init_30_30.csv'
    }

if __name__ == "__main__":
    mode = 'test'  # Assuming you're testing, adjust as necessary for other modes
    args = get_default_args()  # Ensure all needed parameters are set correctly

    # Initialize dataset
    try:
        dataset = RELLIS_3D(mode, args)
        print('Dataset initialized successfully.')
    except Exception as e:
        print(f"Error during dataset initialization: {e}")

    # Check dataset contents
    try:
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            print("Sample data check:")
            for i in range(min(5, len(dataset))):  # Print details of up to 5 samples
                print("Sample data check success")
                # print(dataset[i])
        else:
            print("No data available in dataset.")
    except Exception as e:
        print(f"Error accessing dataset items: {e}")

    print('Test complete.')
