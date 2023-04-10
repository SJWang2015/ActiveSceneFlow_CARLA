import os
# import open3d as o3d
import numpy as np 
import re

LIDAR_TO_WORLD_EULER = (np.pi/2,-np.pi/2,0)

def readData(filename):
	"""
	reads the ground truth file 
	returns a 2D array with each 
	row as GT pose(arranged row major form)
	array size should be 1101*12
	"""
	data = np.loadtxt(filename)
	#data[i].reshape(3,4)
	return data 

def readRawData(filename, pattern):
	"""
	reads the ground truth file 
	returns a 2D array with each 
	row as GT pose(arranged row major form)
	array size should be 1101*12
	"""
	with open(filename, 'r') as file:
		log = file.read()
		r = re.findall(pattern, log)[0]
		if len(r) == 6:
			data = [float(r[0]), float(r[1]), float(r[2]), np.radians(float(r[3])), np.radians(float(r[4])), np.radians(float(r[5]))]
			# print(data)
		elif len(r) == 7:
			data = [int(r[0]), float(r[1]), float(r[2]), float(r[3]), np.radians(float(r[4])), np.radians(float(r[5])), np.radians(float(r[6]))]
			# print(data)
		else:
			print("WRONG DATA.")
	#data[i].reshape(3,4)
	return np.array(data)


def readPointCloud(filename):
	"""
	reads bin file and returns
	as m*4 np array
	all points are in meters
	you can filter out points beyond(in x y plane)
	50m for ease of computation
	and above or below 10m
	"""
	pcl = np.fromfile(filename, dtype=np.float32,count=-1)
	pcl = pcl.reshape([-1,4])
	return pcl 


def rotation_from_euler_zyx(alpha, beta, gamma):
    R = np.zeros((3, 3), dtype='double')
    
    R[0, 0] = np.cos(alpha) * np.cos(beta) 
    R[0, 1] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
    R[0, 2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
    
    R[1, 0] = np.sin(alpha) * np.cos(beta)
    R[1, 1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
    R[1, 2] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
    
    R[2, 0] = -np.sin(beta)
    R[2, 1] =  np.cos(beta) * np.sin(gamma)
    R[2, 2] =  np.cos(beta) * np.cos(gamma)
    
    return R

def lidar_to_world(arr: np.ndarray):
	'''
	just a rotation, so done separately
	'''
	if arr.shape[1] != 3:
		raise ValueError(f'arr should have 3 columns, instead got {arr.shape[1]}')

	T = rotation_from_euler_zyx(*LIDAR_TO_WORLD_EULER)
	return (T @ arr.T).T

def make_homogenous_and_transform(arr: np.ndarray, T: np.ndarray):
	if arr.shape[1] != 3 or T.shape != (3, 4):
		raise ValueError(f'arr should have 3 columns and T should be 3x4, instead got {arr.shape[1]} and {T.shape}')

	T_4 = np.vstack((T, [0, 0, 0, 1]))
	arr_4 = np.c_[arr, np.ones(arr.shape[0])]

	transformed_arr = T_4 @ arr_4.T
	return transformed_arr.T[:, :-1]


class Rotation:
    def __init__(self, roll, pitch, yaw):
		# pitch: (float – degrees) – Y-axis rotation angle.
		# yaw: (float – degrees) – Z-axis rotation angle.
		# roll: (float – degrees) – X-axis rotation angle.
		
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
    
    def RotateVector(self, v):
        cy = np.cos(self.yaw)
        sy = np.sin(self.yaw)

        cr = np.cos(self.roll)
        sr = np.sin(self.roll)

        cp = np.cos(self.pitch)
        sp = np.sin(self.pitch)
        x = v[:,0]
        y = v[:,1]
        z = v[:,2]



        out_point_x = x * (cp * cy) + y * (cy * sp * sr - sy * cr) + z * (-cy * sp * cr - sy * sr)
        out_point_y = x * (cp * sy) + y * (sy * sp * sr + cy * cr) + z * (-sy * sp * cr + cy * sr)
        out_point_z = x * (sp) + y * (-cp * sr) + z * (cp * cr)
        out_point = np.array([out_point_x, out_point_y, out_point_z])
        return out_point

    def InverseRotateVector(self, v):
        cy = np.cos(self.yaw)
        sy = np.sin(self.yaw)

        cr = np.cos(self.roll)
        sr = np.sin(self.roll)

        cp = np.cos(self.pitch)
        sp = np.sin(self.pitch)
        x = v[:,0]
        y = v[:,1]
        z = v[:,2]

        out_point_x = x * (cp * cy) + y * (cp * sy) + z * (sp)
        out_point_y = x * (cy * sp * sr - sy * cr) + y * (sy * sp * sr + cy * cr) + z * (-cp * sr)
        out_point_z = x * (-cy * sp * cr - sy * sr) + y * (-sy * sp * cr + cy * sr) + z * (cp * cr)
        out_point = np.array([out_point_x, out_point_y, out_point_z]).T

        return out_point


def GetMatrix(yaw, roll, pitch, x, y, z):
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    transform = np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
    [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
    [sp, -cp * sr, cp * cr, z],
    [0.0, 0.0, 0.0, 1.0]])

    return transform

def GetInverseMatrix(yaw, roll, pitch, x, y, z):
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    a = np.zeros(1,3)
    a = InverseTransformPoint(a)

    transform = np.array([[ cp * cy, cp * sy, sp, a[0]],
    [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, -cp * sr, a[1]],
    [-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr, a[2]],
    [0.0, 0.0, 0.0, 1.0]])

    return transform

def TransformPoint(v, loc, rot_c):
    out_point = rot_c.RotateVector(v)
    out_point += loc
    return out_point.T

def InverseTransformPoint(v, loc, rot_c):
    v -= loc
    out_point = rot_c.InverseRotateVector(v)
    return out_point

