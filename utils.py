import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
from yaml.loader import SafeLoader
import cv2
from scipy.interpolate import griddata
from config import IMAGE_HEIGHT, IMAGE_WIDTH

def georeference_point_eq(intrinsic_matrix: np.ndarray,
                            image_points: np.ndarray,
                            camer_rotation: np.ndarray,
                            translation: np.ndarray,
                            target_elevation: np.ndarray) -> np.ndarray:
    """
    Estimate origin of pixel point using georeferencing.
    Relies on as of yet unported ROS2 functionality (transform).
    :param Header header: Header with timestamp
    :param np.ndarray image_points: Image point to georeference
    :param str camera: Origin camera
    :param np.ndarray ownship_elevation: Elevation of ownship
    :return np.ndarray: Cartesian estimate of pixel origin
    """
    intrinsic_matrix = intrinsic_matrix

    rot_mat = R.from_euler('xyz', camer_rotation).as_matrix().T
    t_vec= -rot_mat@np.transpose(translation)
    extrinsic_matrix = np.concatenate((rot_mat, t_vec), axis=1) #This is kind of invR|-invR*t
    
    P = intrinsic_matrix @ extrinsic_matrix

    x_p, y_p = image_points
    # Calculate coefficients for the left/right side of reverse pinhole model
    left_side = np.array(
        [[x_p*P[2, 0] - P[0, 0], x_p*P[2, 1] - P[0, 1]],
            [y_p*P[2, 0] - P[1, 0], y_p*P[2, 1] - P[1, 1]]])

    right_side = np.array(
        [[target_elevation*(P[0, 2]-x_p*P[2, 2])+P[0, 3]-x_p*P[2, 3]],
            [target_elevation*(P[1, 2]-y_p*P[2, 2])+P[1, 3]-y_p*P[2, 3]]])

    xy = np.linalg.inv(left_side)@right_side

    pos_estimate = np.array(
        [xy[0, 0],
            xy[1, 0],
            target_elevation])
    return pos_estimate 


class Image:
    def __init__(self,
                 im: np.array,
                 K: np.array,
                 D: np.array,
                 camera_translation: np.array,
                 camera_rotation: np.array
                 ):

        self._im = im

        self._height = self._im.shape[0]
        self._width = self._im.shape[1]
        self._chan = self._im.shape[2]
        self._row_cut_off = 350


        self._im_undistorted = cv2.undistort(self._im, K, D)
        self._im_pos = self.calculate_im_pos(K, camera_rotation, camera_translation)
        self._im_pos_cut = self.cut_im_pos()
        self._im_undistorted_cut, self._im_bird_eye = self.calculate_new_image(camera_rotation)


    def calculate_im_pos(self, K, camera_rotation, camera_translation):
        target_elevation = np.array(0)  

        xx,yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        grid = np.array([xx,yy])
        grid = np.einsum('ijk->jki', grid)

        im_pos = np.zeros((self.height, self.width,3), dtype=np.float32)

        for row in grid:
            for coor in row:
                pixel = coor
                pos = georeference_point_eq(K, pixel, camera_rotation, camera_translation, target_elevation).astype(np.float32)
                im_pos[int(pixel[1]),int(pixel[0])] = np.concatenate((pos[0:2],np.array([1])))
    
        # np.save('Camera_FR_distance_array.npy', im_pos)     
        # im_pos = np.load('Camera_FR_distance_array.npy')
        return im_pos

    def cut_im_pos(self):

        cut_off_distance = 20
        im_pos_cut = self.im_pos.copy()[self.row_cut_off:,:,:]

        mask = np.any((abs(im_pos_cut) > cut_off_distance), axis=2)
        im_pos_cut[mask] = np.array([np.nan, np.nan, np.nan])

        return im_pos_cut
    
    def rotate_im_pos_from_original_to_forward(self, camera_rotation, im_pos_cut):
        im_pos_cut = im_pos_cut
        a = camera_rotation[2] - math.pi/2
        b = 2*math.pi - abs(camera_rotation[2] - math.pi/2)
        yaw_rotation_to_forward = np.min(np.array([a, b]))
        rot_mat = R.from_euler('xyz', np.array([0, 0, yaw_rotation_to_forward])).as_matrix()

        im_pos_cut_F = np.einsum('ij,jkl->ikl', rot_mat,im_pos_cut)

        return im_pos_cut_F


    def rotate_im_pos_from_forward_to_original(self, camera_rotation, im_pos_cut_F):
        im_pos_cut_F = im_pos_cut_F

        a = camera_rotation[2] - math.pi/2
        b = 2*math.pi - abs(camera_rotation[2] - math.pi/2)
        yaw_rotation_to_forward = np.min(np.array([a, b]))
        rot_mat = R.from_euler('xyz', np.array([0, 0, yaw_rotation_to_forward])).as_matrix().T

        im_pos_original_camera = np.einsum('ij,jkl->ikl', rot_mat, im_pos_cut_F)

        return im_pos_original_camera  
    
    def normalize_im_pos(self, camera_rotation, im_pos_cut_t):
        im_pos_cut_F = im_pos_cut_t
        # im_pos_cut_F = self.rotate_im_pos_from_original_to_forward(camera_rotation, im_pos_cut)

        switch_xy = np.array([[0,1,0],
                              [1,0,0],
                              [0,0,1]])


        im_pos_cut_F = np.einsum('ij, jkl->ikl', switch_xy, im_pos_cut_F)
        # im_pos_cut_F =  im_pos_cut_t = np.einsum('ijk->jki',im_pos_cut_F)





        max_x = np.nanmax(im_pos_cut_F[0,:,:])
        # min_x_index = np.where(im_pos_cut[:,:,0] == min_y)[-1] 
        # min_x = im_pos_cut[-1,min_x_index[-1]][1]
        min_x = np.nanmin(im_pos_cut_F[0,:,:])

        # Swithing from x forward, y right axis to x right, y forward to match image cooridnates
        max_y = np.nanmax(im_pos_cut_F[1,:,:])
        # min_y = np.nanmin(np.ma.masked_array(im_pos_cut[:,:,0], mask=im_pos_cut[:,:,0]==0))
        min_y = np.nanmin(im_pos_cut_F[1,:,:])

        # # Assuming mirrored values on negative side of y axis
        # max_x = np.nanmax(im_pos_cut_F[:,:,0])
        # # min_x_index = np.where(im_pos_cut[:,:,0] == min_y)[-1] 
        # # min_x = im_pos_cut[-1,min_x_index[-1]][1]
        # min_x = np.nanmin(im_pos_cut_F[:,:,0])

        # Normalize im_pos to origo in middle and -1 to 1 on axis

        s_x = 2/(max_x-min_x)
        t_x = -s_x * min_x-1

        s_y = 2/(max_y-min_y)
        t_y = -s_y * min_y-1

        N = np.array([[s_x, 0,  t_x],
                      [0, s_y,  t_y],
                      [0, 0,    1]])
        im_pos_cut_F_normalized = np.einsum('ij,jkl->ikl', N, im_pos_cut_F)

        norm_x_max_F = np.nanmax(im_pos_cut_F_normalized[0,:,:])
        norm_x_min_F = np.nanmin(im_pos_cut_F_normalized[0,:,:])
        norm_y_max_F = np.nanmax(im_pos_cut_F_normalized[1,:,:])
        norm_y_min_F = np.nanmin(im_pos_cut_F_normalized[1,:,:])

        # im_pos_cut_normalized = self.rotate_im_pos_from_forward_to_original(camera_rotation, im_pos_cut_F_normalized)

        # norm_x_max = np.nanmax(im_pos_cut_normalized[0,:,:])
        # norm_x_min = np.nanmin(im_pos_cut_normalized[0,:,:])
        # norm_y_max = np.nanmax(im_pos_cut_normalized[1,:,:])
        # norm_y_min = np.nanmin(im_pos_cut_normalized[1,:,:])

        return im_pos_cut_F_normalized

    def calculate_new_image(self, camera_rotation):

        im_pos_cut = self.im_pos_cut
        im_pos_cut_t = np.einsum('ijk->kij',im_pos_cut)
        
        im_cut = self.im_undistorted.copy()[self.row_cut_off:,:,:]
        im_pos_normalized = self.normalize_im_pos(camera_rotation,im_pos_cut_t)


        new_h, new_w = IMAGE_HEIGHT, IMAGE_WIDTH

        K = np.array([[new_w/2-1,     0,          new_w/2],
                      [0,           -(new_h/2-1),    new_h/2],
                      [0,           0,          1]])

        im_pos_pixel = np.einsum('ij,jkl->ikl', K, im_pos_normalized)
        im_pos_pixel = np.einsum('ijk->jki',im_pos_pixel)

        im_pos_pixel =  np.nan_to_num(im_pos_pixel, nan = 99999999)
        im_pos_pixel = im_pos_pixel.astype(int)
        im_pos_pixel = im_pos_pixel[:,:,:2]

        points_x_all = im_pos_pixel[:,:,1]
        points_x = np.transpose(np.array([np.ravel(points_x_all)]))
        points_x_all = np.transpose(np.array([np.ravel(points_x_all)]))
        points_x = np.transpose(np.array([points_x[points_x != 99999999]]))

        points_y = im_pos_pixel[:,:,0]
        points_y = np.transpose(np.array([np.ravel(points_y)]))
        points_y = np.transpose(np.array([points_y[points_y != 99999999]]))

        # points_y = np.transpose(np.array([np.ravel(im_pos_pixel[im_pos_pixel[:,:,0] != 99999999])]))
        points = np.concatenate((points_x,points_y), axis=1)
        grid_x,grid_y = np.meshgrid(range(new_h), range(new_w), indexing='ij')

        # mask = np.any(> cut_off_distance), axis=2)
        rgb = im_cut

        rgb = np.reshape(rgb,(len(points_x_all), self._chan))
        rgb = np.delete(rgb, np.where(points_x_all == 99999999), axis=0)

        grid_z0 = griddata(points, rgb, (grid_x, grid_y), method='linear')
        grid_z0[np.where(np.isnan(grid_z0))] = 0
        grid_z0 = grid_z0[:,:,:].astype(np.uint8)
        


        # im_bird_eye = np.zeros((new_h,new_w,3)).astype(int)
        # im_cut_h, im_cut_w,_ = np.shape(im_cut)
        # for i in range(im_cut_h-1):
        #     for j in range(im_cut_w-1):
        #         pixel = im_pos_pixel[i,j,:]
        #         if ((pixel[1]>new_h) or (pixel[0]>new_w)):
        #             continue
        #         rgb = im_cut[i,j,:]
        #         im_bird_eye[pixel[1],pixel[0],:] = rgb
        return im_cut, grid_z0
     

    def update_im(self, im: np.array, K: np.array, D: np.array, camera_rotation: np.array):
        self._im = im
        self._im_undistorted = cv2.undistort(self._im, K, D)
        self._im_undistorted_cut, self._im_bird_eye = self.calculate_new_image(camera_rotation)

    @property
    def im(self):
        return self._im
    @property
    def im_undistorted(self):
        return self._im_undistorted
    @property
    def im_pos(self):
        return self._im_pos
    @property
    def width(self):
        return self._width 
    @property
    def height(self):
        return self._height 
    @property
    def chan(self):
        return self._chan
    @property
    def im_bird_eye(self):
        return self._im_bird_eye
    @property
    def im_undistorted_cut(self):
        return self._im_undistorted_cut
    @property
    def row_cut_off(self):
        return self._row_cut_off
    @property
    def im_pos_cut(self):
        return self._im_pos_cut
        

def get_camera_name_from_topic_str(topic_str: str):
    """Returns the camera name from the topic string"""
    indices = [i for i, c in enumerate(topic_str) if c == '/']

    start_index = indices[2]+1
    end_index = indices[3]
    name = topic_str[start_index:end_index]
    return name

class Camera:
    def __init__(self, msg, topic: str):
        self._name = get_camera_name_from_topic_str(topic)
        self._D = np.array([msg.D])
        self._K = np.reshape(np.array([msg.K]), (3,3))
        self._P = np.reshape(np.array([msg.P]), (3,4))
        self._R = np.reshape(np.array([msg.R]), (3,3))
        self._topic = topic

        self._im = None
        self._camera_rotation = None
        self._camera_translation = None

    def update_camera_roation_and_translation(self, dict):
        camera_key = 'eo_'+ self.name.lower()

        rotation = np.array(dict[camera_key]['static_transform']['rotation'])
        translation = np.array([dict[camera_key]['static_transform']['translation']])

        self._camera_rotation = rotation
        self._camera_translation = translation

    def update_image_in_camera_cls(self, im_rgb: np.array):
        if (self.im is None):
            im = Image(im_rgb,self.K, self.D, self.camera_translation, self.camera_rotation)
            self._im = im
        else:
            self._im.update_im(im_rgb,self.K, self.D, self.camera_rotation)

    @property
    def name(self):
        return self._name
    @property
    def D(self):
        return self._D
    @property
    def K(self):
        return self._K
    @property
    def P(self):
        return self._P
    @property
    def topic(self):
        return self._topic
    @property
    def im(self):
        return self._im
    @property
    def camera_rotation(self):
        return self._camera_rotation
    @property
    def camera_translation(self):
        return self._camera_translation

class BirdsEyeView:
    def __init__(self, im_F:np.array, im_FR:np.array, im_FL:np.array, im_RR:np.array, im_RL:np.array, im_mA1: np.array):
        self._im_F = im_F
        self._im_FR = im_FR
        self._im_FL = im_FL
        self._im_RR = im_RR
        self._im_RL = im_RL
        self._mA1 = im_mA1
        self._birds_eye = self.make_birds_eye()

    def make_birds_eye(self):
        offset_w = 400
        offset_h = 550
        birds_eye_width = IMAGE_WIDTH*2+offset_w
        birds_eye_height = IMAGE_HEIGHT*2 +offset_h
        mA1_width = 700
        mA1_heigth = 900
        im_mA1 = cv2.resize(self._mA1,(mA1_width,mA1_heigth))

        im_birds_eye = np.zeros((birds_eye_height, birds_eye_width,3), dtype=np.uint8)
        # Insert small images into larger
        im_birds_eye[offset_h:(offset_h+IMAGE_HEIGHT),offset_w+IMAGE_WIDTH:,:] = self._im_FR
        im_birds_eye[offset_h:(offset_h+IMAGE_HEIGHT),:IMAGE_WIDTH,:] = self._im_FL
        im_birds_eye[(offset_h+IMAGE_HEIGHT):,offset_w+IMAGE_WIDTH:,:] = self._im_RR
        im_birds_eye[(offset_h+IMAGE_HEIGHT):, :IMAGE_WIDTH,:] = self._im_RL
        im_birds_eye[:IMAGE_HEIGHT,(birds_eye_width//2-IMAGE_WIDTH//2):(birds_eye_width//2+IMAGE_WIDTH//2)] += self._im_F
        im_birds_eye[(birds_eye_height//2-mA1_heigth//2):(birds_eye_height//2+mA1_heigth//2),
                     (birds_eye_width//2-mA1_width//2):(birds_eye_width//2+mA1_width//2),:] = im_mA1
        mask = np.any(im_birds_eye[:,:,:] == 0, axis=2)
        im_birds_eye[mask] = np.array([255, 255, 255])

        return im_birds_eye

    @property
    def birds_eye(self):
        return self._birds_eye

class VesselmA1:    

    def __init__(self, num_cameras: int):
        self._number_of_cameras = num_cameras
        self._F = None
        self._FR = None
        self._FL = None
        self._RR = None
        self._RL = None

    def set_camera_variables(self, camera: Camera):
        camera_name = camera.name
        if(camera_name =='F'):
            self._F = camera
        elif(camera_name =='FR'):
            self._FR = camera
        elif(camera_name =='FL'):
            self._FL = camera
        elif(camera_name =='RR'):
            self._RR = camera
        elif(camera_name =='RL'):
            self._RL = camera
    
    def init_completed(self):
        if(None in (self.F, self.FR, self.FL, self.RR, self.RL)):
            return False
        return True

    @property
    def F(self):
        return self._F
    @property
    def FR(self):
        return self._FR
    @property
    def FL(self):
        return self._FL
    @property
    def RR(self):
        return self._RR
    @property
    def RL(self):
        return self._RL


def yaml_file_to_dict(file_name: str):
    with open(file_name) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data
