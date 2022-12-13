import time
import numpy as np
import cv2
from config import BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH
import rosbag
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from utils import VesselmA1, Camera, make_BEW, yaml_file_to_dict, get_camera_name_from_topic_str



def init():
    number_of_cameras = 5
    yaml_file_name = 'camer_pos.yaml'
    camera_positions = yaml_file_to_dict(yaml_file_name)
    bag_name = "/home/mathias/Documents/Project thesis/2021-05-05/2021-05-05/2021-05-05-11-30-20.bag"
    vesselmA1 = VesselmA1(number_of_cameras)
    bag = rosbag.Bag(bag_name,'r')

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
    print('Init complete')

    return bag, vesselmA1




def main():
    st = time.time()

    bag, vesselmA1 = init()
    camera_check = np.zeros((10000,6))
    bridge = CvBridge()
    count = 1
    frame = 1
    file_path = "Videos/BEW_3.mp4"
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(file_path, fourcc, 5, (BEW_IMAGE_WIDTH, BEW_IMAGE_HEIGHT))

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
            camera_check[frame][0] = 1


        elif(camera_name =='FR'):
            print('FR')
            vesselmA1._FR.update_image_in_camera_cls(im_rgb)
            camera_check[frame][1] = 1
            
        elif(camera_name =='FL'):
            print('FL')
            vesselmA1._FL.update_image_in_camera_cls(im_rgb)
            camera_check[frame][2] = 1

        elif(camera_name =='RR'):
            print('RR')
            vesselmA1._RR.update_image_in_camera_cls(im_rgb)
            camera_check[frame][3] = 1

        elif(camera_name =='RL'):
            print('RL')
            vesselmA1._RL.update_image_in_camera_cls(im_rgb)
            camera_check[frame][4] = 1
        
        print("Wrote image %i" % count)
        # if ((count//5*40 > 14390) & (count//5*40 < 14410)):
        #     if count % (5) == 0:
        #         print("Making bird eye view")
        #         im_BEW = make_BEW(vesselmA1)
        #         plt.figure()
        #         plt.imshow(im_BEW)
        #         plt.show()
        #         plt.imsave('im_BEW'+str(count//5)+'.jpg', im_BEW)
        #         plt.imsave('im_F'+str(count)+'.jpg', vesselmA1.F.im.im)
        #         plt.imsave('im_FR'+str(count)+'.jpg', vesselmA1.FR.im.im)
        #         plt.imsave('im_FL'+str(count)+'.jpg', vesselmA1.FL.im.im)
        #         plt.imsave('im_RR'+str(count)+'.jpg', vesselmA1.RR.im.im)
        #         plt.imsave('im_RL'+str(count)+'.jpg', vesselmA1.RL.im.im)


        # if (count//5*40 > 14411):
        #     break
        
        
            
            
        if (count % (5) == 0) & (count>10):
            camera_check[frame][5] = np.sum(camera_check[frame,:5])
            frame += 1
            print("Making bird eye view")
            im_BEW = make_BEW(vesselmA1)
            writer.write(cv2.cvtColor(im_BEW, cv2.COLOR_BGR2RGB))

            et = time.time()
            elapsed_time = et - st
            print('Execution time:', elapsed_time, 'seconds')
            # im_name = "360bew_crude_stitch_"+str(count//5*40)+".jpg"
            # plt.imsave(im_name, im_BEW)
            # plt.figure()
            # plt.imshow(im_BEW)
            # plt.show()
        # if count > (5*5):
        #     break
        
        count += 1
    writer.release()  
    bag.close()
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')




if __name__ == '__main__':
    main()