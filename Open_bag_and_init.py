import numpy as np
import cv2
import rosbag
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from utils import VesselmA1, Camera, make_BEW, yaml_file_to_dict, get_camera_name_from_topic_str



def init():
    number_of_cameras = 5
    yaml_file_name = 'camer_pos.yaml'
    camera_positions = yaml_file_to_dict(yaml_file_name)

    vesselmA1 = VesselmA1(number_of_cameras)
    bag = rosbag.Bag("2021-05-05/2021-05-05/2021-05-05-10-58-01.bag",'r')

    for(topic, msg, t) in  bag.read_messages(topics=['/sensor_rig/optical/F/camera_info', 
                                                    '/sensor_rig/optical/FR/camera_info',
                                                    '/sensor_rig/optical/FL/camera_info',
                                                    '/sensor_rig/optical/RR/camera_info',
                                                    '/sensor_rig/optical/RL/camera_info']):
        if(vesselmA1.init_completed()):
            break
        camera = Camera(msg,topic)
        camera.update_camera_roation_and_translation(camera_positions)
        vesselmA1.set_camera_variables(camera)

    return bag, vesselmA1




def main():

    bag, vesselmA1 = init()
    
    bridge = CvBridge()
    count = 1


    for topic, msg, t in bag.read_messages(topics=[ '/sensor_rig/optical/F/image_raw',
                                                    '/sensor_rig/optical/FR/image_raw',
                                                    '/sensor_rig/optical/FL/image_raw',
                                                    '/sensor_rig/optical/RR/image_raw',
                                                    '/sensor_rig/optical/RL/image_raw']):
        im_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        # im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        
        camera_name = get_camera_name_from_topic_str(topic)
        if(camera_name =='F'):
            print('F')
            vesselmA1._F.update_image_in_camera_cls(im_rgb)
            # plt.figure()
            # plt.imshow(im_rgb)
            # plt.figure()
            # plt.imshow(vesselmA1.F.im.im_undistorted_cut)
            # plt.figure()
            # plt.imshow(vesselmA1.F.im.im_bird_eye)
            # plt.imsave('F.jpg',vesselmA1.F.im.im_bird_eye)

            print('Next camera coming up!')
        elif(camera_name =='FR'):
            print('FR')
            vesselmA1._FR.update_image_in_camera_cls(im_rgb)
            # plt.figure()
            # plt.imshow(im_rgb)
            # plt.figure()
            # plt.imshow(vesselmA1.FR.im.im_undistorted_cut)
            # plt.figure()
            # plt.imshow(vesselmA1.FR.im.im_bird_eye)
            # plt.imsave('FR.jpg', vesselmA1.FR.im.im_bird_eye)

            print('Next camera coming up!')
            
        elif(camera_name =='FL'):
            print('FL')
            vesselmA1._FL.update_image_in_camera_cls(im_rgb)
            # plt.figure()
            # plt.imshow(im_rgb)
            # plt.figure()
            # plt.imshow(vesselmA1.FL.im.im_undistorted_cut)
            # plt.figure()
            # plt.imshow(vesselmA1.FL.im.im_bird_eye)
            # plt.imsave('FL.jpg', vesselmA1.FL.im.im_bird_eye)

            print('Next camera coming up!')

        elif(camera_name =='RR'):
            print('RR')
            vesselmA1._RR.update_image_in_camera_cls(im_rgb)
            # plt.figure()
            # plt.imshow(im_rgb)
            # plt.figure()
            # plt.imshow(vesselmA1.RR.im.im_undistorted_cut)
            # plt.figure()
            # plt.imshow(vesselmA1.RR.im.im_bird_eye)
            # plt.imsave('RR.jpg', vesselmA1.RR.im.im_bird_eye)
            print('Next camera coming up!')
        elif(camera_name =='RL'):
            print('RL')
            vesselmA1._RL.update_image_in_camera_cls(im_rgb)
            # plt.figure()
            # plt.imshow(im_rgb)
            # plt.figure()
            # plt.imshow(vesselmA1.RL.im.im_undistorted_cut)
            # plt.figure()
            # plt.imshow(vesselmA1.RL.im.im_bird_eye)
            # plt.imsave('RL.jpg', vesselmA1.RL.im.im_bird_eye)

            print('Next camera coming up!')
        
        print("Wrote image %i" % count)
        if count % (5*40) == 0:
            im_BEW = make_BEW(vesselmA1)
            im_name = "360bew_crude_stitch_"+str(count//5*40)+".jpg"
            plt.imsave(im_name, im_BEW)
            plt.figure()
            plt.imshow(im_BEW)
            plt.show()
            # break
            # plt.show()
            # cv2.waitKey(0)

            
        #     plt.figure()
        #     plt.imshow(vesselmA1.F.im.im_bird_eye)

        #     plt.figure()
        #     plt.imshow(vesselmA1.FL.im.im_bird_eye)

        #     plt.figure()
        #     plt.imshow(vesselmA1.FR.im.im_bird_eye)

        #     plt.figure()
        #     plt.imshow(vesselmA1.RL.im.im_bird_eye)

        #     plt.figure()
        #     plt.imshow(vesselmA1.RR.im.im_bird_eye)
            

        #     plt.show()
        #     cv2.waitKey(20)

        count += 1  
    plt.show()
    bag.close()


if __name__ == '__main__':
    main()