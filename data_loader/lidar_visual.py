import numpy as np
import cv2

def load_point_cloud(bin_path):
    """ 포인트 클라우드 데이터 로딩 """
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud

def project_to_image(points, P, Tr):
    """ 포인트 클라우드를 이미지에 투영 """
    # 포인트 클라우드를 homogeneous 좌표계로 변환
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points[:, :3], ones))
    
    # 라이다 좌표에서 카메라 좌표로 변환
    points_cam = Tr @ points_hom
    
    # 카메라 좌표에서 이미지 좌표로 투영
    points_img = P @ points_cam
    points_img /= points_img[2, :]  # 균일 좌표계로 나눔
    
    return points_img[:2, :].T

def draw_lidar_on_image(img, points, colors):
    """ 이미지에 라이다 데이터 그리기 """
    for point, color in zip(points, colors):
        x, y = int(point[0]), int(point[1])
        # color 정보를 튜플로 변환합니다. OpenCV는 BGR 순서를 사용합니다.
        # color 배열의 형태를 확인하고, 필요에 따라 인덱싱을 조정합니다.
        if color.ndim > 1:
            color = color[0]  # color가 (1, 3) 형태인 경우, 첫 번째 요소를 사용
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))
        if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
            cv2.circle(img, (x, y), 2, color_tuple, -1)

    return img



# 사용 예
bin_path = '/Users/baekchaeyoon/Desktop/Univ/VIP_Lab/RELLIS_data/data/RELLIS-3D/RELLIS-3D/00000/os1_cloud_node_kitti_bin/000000.bin'
image_path = '/Users/baekchaeyoon/Desktop/Univ/VIP_Lab/RELLIS_data/data/RELLIS-3D/RELLIS-3D/00000/pylon_camera_node/frame000000-1581624652_750.jpg'

Tr = np.array([[ 0.03462247,  0.99936055, -0.00893175, -0.03566209],
 [ 0.00479177, -0.00910301, -0.99994709, -0.17154603],
 [-0.99938898,  0.03457784, -0.00510388, -0.13379309],
 [ 0.,          0.,          0.,          1.        ]])

P = np.array([[2.34470273e+03, 0.00000000e+00, 8.07738143e+02, 0.00000000e+00],
 [0.00000000e+00, 2.10624456e+03, 4.68037479e+02, 0.00000000e+00],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# 데이터 로드
point_cloud = load_point_cloud(bin_path)
image = cv2.imread(image_path)

# 포인트 클라우드 투영
projected_points = project_to_image(point_cloud[:, :3], P, Tr)

# 인텐시티를 색상으로 변환
# 인텐시티를 색상으로 변환
intensities = (point_cloud[:, 3] * 255).astype(np.uint8)  # intensity를 0-255 범위로 스케일
colors = cv2.applyColorMap(intensities, cv2.COLORMAP_HOT)

# 이미지에 포인트 클라우드 그리기
overlay_image = draw_lidar_on_image(image, projected_points, colors)

# 결과 이미지 저장 또는 표시
cv2.imwrite('overlayed_image.jpg', overlay_image)
