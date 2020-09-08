'''
Using the data of obstacle instead of using the camera detection now
Using the position data of the sensors instead of a computed number
'''
import glob
import sys
import icp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from obstacle_detect.Run import *
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc
import cv2
import nvector as nv
import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_TAB
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
    # np.set_printoptions(threshold=sys.maxsize)
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 540

# FOV (field of view): the extent of the observable world seen at any given moment.
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)
saved = False


class ClientSideBoundingBox(object):

    @staticmethod
    def get_bounding_boxes(obstacles, camera):
        bounding_boxes = [ClientSideBoundingBox.get_bounding_box(obstacle, camera) for obstacle in obstacles]
        bounding_boxes = [bounding_box for bounding_box in bounding_boxes if all(bounding_box[:,2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display,bounding_boxes):
        bounding_box_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bounding_box_surface.set_colorkey((0,0,0))
        for bounding_box in bounding_boxes:
            points = [(int(bounding_box[i,0]), int(bounding_box[i,1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[0], points[1])
            # pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bounding_box_surface, BB_COLOR, points[3], points[7])
        display.blit(bounding_box_surface, (0, 0))

    @staticmethod
    def create_bounding_area_masks(bounding_boxes, depth_img):

        masks = []
        for bounding_box in bounding_boxes:
            mask = np.ones((VIEW_HEIGHT, VIEW_WIDTH), np.uint8)*255
            points = [(int(bounding_box[i, 0]),int(bounding_box[i, 1])) for i in range(8)]

            cv2.line(mask, points[0], points[1], 0, thickness=2)
            cv2.line(mask, points[1], points[2], 0, thickness=2)
            cv2.line(mask, points[2], points[3], 0, thickness=2)
            cv2.line(mask, points[3], points[0], 0, thickness=2)
            cv2.line(mask, points[4], points[5], 0, thickness=2)
            cv2.line(mask, points[5], points[6], 0, thickness=2)
            cv2.line(mask, points[6], points[7], 0, thickness=2)
            cv2.line(mask, points[7], points[4], 0, thickness=2)
            cv2.line(mask, points[0], points[4], 0, thickness=2)
            cv2.line(mask, points[1], points[5], 0, thickness=2)
            cv2.line(mask, points[2], points[6], 0, thickness=2)
            cv2.line(mask, points[3], points[7], 0, thickness=2)

            retval, labels = cv2.connectedComponents(mask)
            bg_label = labels[0, 0]
            mask = np.ones((VIEW_HEIGHT, VIEW_WIDTH), np.uint8)
            # print(np.shape(mask))
            mask[labels == bg_label] = 0
            '''
            min_dist = np.min(depth_img[mask != 0])
            for i in range(5, VIEW_HEIGHT-5):
                for j in range(5, VIEW_WIDTH-8):
                    if depth_img[i, j] == 50 and mask[i, j] != 0:
                        times = 0
                        diff = 0.5
                        for ii in range(i-4, i+4):
                            for jj in range(j-3, j+3):
                                if 0 < abs(depth_img[ii, jj]-min_dist) < diff:
                                    times += 1
                        if times < 2:
                            mask[i, j] = 0
                    if (mask[i, j] != 0) and (depth_img[i, j] < min_dist-0.5 or depth_img[i, j] > min_dist+0.5):
                        if depth_img[i, j] != 50:
                            mask[i, j] = 0
            '''
            masks += [mask]
            # print(bg_label)

        return masks

    @staticmethod
    def get_bounding_box(obstacle,camera):
        bounding_box_cords = ClientSideBoundingBox.create_bounding_box_points(obstacle)
        # vehicle reference
        cords_x_y_z = ClientSideBoundingBox.vehicle_to_camera(bounding_box_cords, obstacle, camera)[:3, :]
        # print(cords_x_y_z)
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        # print(cords_y_minus_z_x)
        bounding_box = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        # camera image reference (u,v,z)

        camera_bounding_box = np.concatenate([bounding_box[:, 0] / bounding_box[:, 2], bounding_box[:, 1] / bounding_box[:, 2],
                                              bounding_box[:, 2]], axis=1)
        # print(camera_bounding_box)
        return camera_bounding_box

    @staticmethod
    def create_bounding_box_points(obstacle):
        cords = np.zeros((8, 4))
        extent = obstacle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def get_matrix(transform):
        #from carla transform to matrix, the matrix contains the location and the rotation matrix
        rotation = transform.rotation
        location = transform.location
        cos_yaw = np.cos(np.radians(rotation.yaw))
        sin_yaw = np.sin(np.radians(rotation.yaw))
        cos_roll = np.cos(np.radians(rotation.roll))
        sin_roll = np.sin(np.radians(rotation.roll))
        cos_pitch = np.cos(np.radians(rotation.pitch))
        sin_pitch = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = cos_pitch*cos_yaw
        matrix[0, 1] = cos_yaw*sin_pitch*sin_roll-sin_yaw*cos_roll
        matrix[0, 2] = -cos_yaw*sin_pitch*cos_roll-sin_yaw*sin_roll
        matrix[1, 0] = sin_yaw*cos_pitch
        matrix[1, 1] = sin_yaw*sin_pitch*sin_roll+cos_yaw*cos_roll
        matrix[1, 2] = -sin_yaw*sin_pitch*cos_roll+cos_yaw*sin_roll
        matrix[2, 0] = sin_pitch
        matrix[2, 1] = -cos_pitch*sin_roll
        matrix[2, 2] = cos_pitch*cos_roll
        return matrix

    @staticmethod
    def vehicle_to_world(cords, obstacle):
        # boundingbox's coordinate in the obstacle's reference
        bounding_box_transform = carla.Transform(obstacle.bounding_box.location)
        bounding_box_obstacle_matrix = ClientSideBoundingBox.get_matrix(bounding_box_transform)

        # obstacle's coordinate in the world reference
        obstacle_world_matrix = ClientSideBoundingBox.get_matrix(obstacle.get_transform())
        bounding_box_world_matrix = np.dot(obstacle_world_matrix, bounding_box_obstacle_matrix)
        bounding_box_world_cords = np.dot(bounding_box_world_matrix, np.transpose(cords))
        return bounding_box_world_cords

    @staticmethod
    def world_to_camera(cords, camera):
        camera_world_matrix = ClientSideBoundingBox.get_matrix(camera.get_transform())
        world_camera_matrix = np.linalg.inv(camera_world_matrix)
        camera_cords = np.dot(world_camera_matrix, cords)
        return camera_cords

    @staticmethod
    def vehicle_to_camera(cords, obstacle, camera):

        world_cord = ClientSideBoundingBox.vehicle_to_world(cords, obstacle)
        camera_cord = ClientSideBoundingBox.world_to_camera(world_cord, camera)
        return camera_cord

    @staticmethod
    def camera1_to_camera2(cords_camera1, camera1, camera2, icp_T):
        """
        Transforms pedestrian coordinates from sensor1 to sensor2.
        """
        camera1_world_matrix = ClientSideBoundingBox.get_matrix(camera1.get_transform())
        # print(camera1_world_matrix)

        camera2_world_matrix = ClientSideBoundingBox.get_matrix(camera2.get_transform())
        # camera2_world_matrix[3, 0] += icp_T[3, 0]
        # camera2_world_matrix[3, 1] += icp_T[3, 1]
        # camera2_world_matrix[3, 2] += icp_T[3, 2]
        # print(camera2_world_matrix)
        world_camera2_matrix = np.linalg.inv(camera2_world_matrix)
        # print(world_camera2_matrix)
        cords_camera2 = world_camera2_matrix.dot(camera1_world_matrix.dot(cords_camera1))
        # print(cords_camera2)
        return cords_camera2

    @staticmethod
    def image1_to_camera(depth_img, masks, camera, dist):

        # Transform the image of sensor1 to the 3D coordinates about sensor1

        if depth_img is None:
            print("depth image broken")
        else:
            width = VIEW_WIDTH
            height = VIEW_HEIGHT
            matrix_camera = np.linalg.inv(camera.calibration)
            cords_x = np.fromfunction(lambda i, j: j*matrix_camera[0, 0]+i*matrix_camera[0, 1], (height, width), dtype=float)
            cords_y = np.fromfunction(lambda i, j: j*matrix_camera[1, 0]+i*matrix_camera[1, 1], (height, width), dtype=float)
            cords_z = np.fromfunction(lambda i, j: j*matrix_camera[2, 0]+i*matrix_camera[2, 1], (height, width), dtype=float)
            cords_all = []
            '''
            TO be fixed here: with overlapping region, how to determine the distance information
            '''
            for mask in masks:
                argmin_dist = np.argmin(depth_img)
                min_dist = np.min(depth_img[mask != 0])
                print(min_dist)
                dist += [min_dist + 0.5]
                # dist += [avg_dist]
                depth_img[mask != 0] = sys.maxunicode

                v0 = argmin_dist//width
                u0 = argmin_dist-v0*width

                x_y_z0 = np.dot(matrix_camera, np.array([u0, v0, 1]).transpose())

                cords = np.stack((cords_x, cords_y, cords_z), axis = -1)-u0*matrix_camera[:, 0]- v0*matrix_camera[:, 1]+x_y_z0

                k = x_y_z0.dot(matrix_camera[:, :2])
                x_y_z0_square = x_y_z0.dot(x_y_z0)
                denominator = np.fromfunction(lambda i,j: k[0]*(j-u0)+k[1]*(i-v0), (height, width), dtype=float)
                denominator += x_y_z0_square
                cords = np.stack((cords[:, :, 0]/denominator, cords[:, :, 1]/denominator, cords[:, :, 2]/denominator), axis=-1)*min_dist*np.sqrt(x_y_z0_square)
                cords_all += [cords]

            return  cords_all

    @staticmethod
    def get_2d_cords_sensor2(rgb_image, depth_img, masks, distances, camera1, camera2, icp_T):
        print(icp_T)
        if (rgb_image is not None) and (depth_img is not None):
            cords_y_minus_z_x_camera1_all = ClientSideBoundingBox.image1_to_camera(depth_img, masks, camera1, distances)
            view_uv_camera2_all = []
            for cords_y_minus_z_x_camera1 in cords_y_minus_z_x_camera1_all:
                cords_y_minus_z_x_camera1 = cords_y_minus_z_x_camera1.reshape(VIEW_HEIGHT*VIEW_WIDTH, 3).transpose()
                cords_camera1 = np.array([cords_y_minus_z_x_camera1[2, :], cords_y_minus_z_x_camera1[0, :],
                                          -cords_y_minus_z_x_camera1[1, :], np.ones(VIEW_HEIGHT*VIEW_WIDTH)])

                cords_camera2 = ClientSideBoundingBox.camera1_to_camera2(cords_camera1, camera1, camera2, icp_T)
                cords_y_minus_z_x_camera2 = np.concatenate([cords_camera2[1, :], -cords_camera2[2, :],
                                                            cords_camera2[0, :]])
                view_uv_camera2 = np.transpose(np.dot(camera2.calibration, cords_y_minus_z_x_camera2))

                view_uv_camera2 = np.concatenate([view_uv_camera2[:, 0]/view_uv_camera2[:, 2],
                                                  view_uv_camera2[:, 1]/view_uv_camera2[:, 2],
                                                  view_uv_camera2[:, 2]], axis=1)

                view_uv_camera2 = np.rint(view_uv_camera2).astype(int)
                view_uv_camera2 = np.array(view_uv_camera2).reshape(VIEW_HEIGHT, VIEW_WIDTH, 3)
                view_uv_camera2_all += [view_uv_camera2]

            return view_uv_camera2_all
        else:
            print("image not prepared")



class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.pedestrian = None
        self.pedestrian2 = None
        self.car_0 = None
        self.car_A = None
        self.car = None
        self.car_controlled = None

        self.display = None
        self.image1 = None
        self.image2 = None
        self.cloud1 = None
        self.cloud2 = None
        self.converted_image = None
        self.depth_image = None
        self.obstacle = None
        self.obstacle_distances = []
        self.view_uv_all = None
        self.pointA = None
        self.point = None
        self.ax = None
        self.ay = None

        self.mask = None
        self.capture_rgb1 = True
        self.capture_rgb2 = True
        self.capture_depth = True
        self.capture_lidar1 = True
        self.capture_lidar2 = True
        self.capture_gnss1 = True
        self.capture_gnss2 = True
        self.saved_file = False
        self.detect_obstacle = True

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        # settings.fixed_delta_seconds = 0.03
        self.world.apply_settings(settings)

    def setup_car_0(self):
        """
        Spawns a car in front of the controlled vehicle.
        """
        car_bp = self.world.get_blueprint_library().filter('vehicle.audi.*')[1]
        # print(car_bp)
        location = self.world.get_map().get_spawn_points()[2]
        location.location.x += 15
        location.location.y -= 0.8
        self.car_0 = self.world.spawn_actor(car_bp, location)

    def setup_pedestrian(self):
        """
        Spawns pedestrians in front of the first vehicle.
        """
        pedestrian_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')[2]
        location = self.world.get_map().get_spawn_points()[2]
        location.location.x += 15
        location.location.y += 0.8
        self.pedestrian = self.world.spawn_actor(pedestrian_bp, location)

    def setup_pedestrian_2(self):
        """
        Spawns pedestrians in front of the first vehicle.
        """
        pedestrian_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')[2]
        location = self.world.get_map().get_spawn_points()[2]
        location.location.x += 17
        location.location.y += 0.9
        self.pedestrian2 = self.world.spawn_actor(pedestrian_bp, location)

    def setup_car_A(self):
        """
        Spawns a car in front of the controlled vehicle.
        """
        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = self.world.get_map().get_spawn_points()[3]
        location.location.x += 8
        location.location.y += 0.8
        self.car_A = self.world.spawn_actor(car_bp, location)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        # location = random.choice(self.world.get_map().get_spawn_points())
        location = self.world.get_map().get_spawn_points()[3]
        location.location.y += 2.8
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        # camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform_A = carla.Transform(carla.Location(x=1.2, y=0, z=1.3), carla.Rotation(pitch=0))
        camera_transform = carla.Transform(carla.Location(x=0.2, y=-0.25, z=1.3), carla.Rotation(pitch=0))
        lidar_transform_A = carla.Transform(carla.Location(x=1.2, y=0, z=2.0), carla.Rotation(pitch=0))
        lidar_transform = carla.Transform(carla.Location(x=0.2, y=-0.25, z=2.0), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        self.camera_A_rgb = self.world.spawn_actor(self.camera_blueprint(), camera_transform_A, attach_to=self.car_A)

        camera_depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        camera_depth_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_depth_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_depth_bp.set_attribute('fov', str(VIEW_FOV))
        self.camera_A_depth = self.world.spawn_actor(camera_depth_bp, camera_transform_A, attach_to=self.car_A)
        '''
        camera_depth_bp2 = self.world.get_blueprint_library().find('sensor.camera.depth')
        camera_depth_bp2.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_depth_bp2.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_depth_bp2.set_attribute('fov', str(VIEW_FOV))
        self.camera_A_depth2 = self.world.spawn_actor(camera_depth_bp2, camera_transform, attach_to=self.car)
        '''
        lidar_bp1 = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp1.set_attribute('range', '5000')
        lidar_bp1.set_attribute('channels', '128')
        lidar_bp1.set_attribute('rotation_frequency', '50')
        lidar_bp1.set_attribute('points_per_second', '1000000')
        self.lidar1 = self.world.spawn_actor(lidar_bp1, lidar_transform_A, attach_to = self.car_A)
        lidar_bp2 = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp2.set_attribute('range', '5000')
        lidar_bp2.set_attribute('channels', '128')
        lidar_bp2.set_attribute('rotation_frequency', '50')
        lidar_bp2.set_attribute('points_per_second', '5000')
        self.lidar2 = self.world.spawn_actor(lidar_bp2, lidar_transform, attach_to = self.car)

        gnss_bp1 = self.world.get_blueprint_library().find('sensor.other.gnss')
        self.gnss1 = self.world.spawn_actor(gnss_bp1, camera_transform_A, attach_to = self.car_A)
        gnss_bp2 = self.world.get_blueprint_library().find('sensor.other.gnss')
        self.gnss2 = self.world.spawn_actor(gnss_bp2, camera_transform, attach_to = self.car)

        weak_self = weakref.ref(self)
        self.camera_A_rgb.listen(lambda image: self.set_rgb_image1(weak_self, image))
        # self.camera_A_depth.listen(lambda image: self.set_depth_image(weak_self, image))
        # self.camera_A_depth2.listen(lambda image: self.set_depth_image2(weak_self, image))
        self.camera.listen(lambda image: self.set_rgb_image2(weak_self, image))
        self.lidar1.listen(lambda cloud: self.set_lidar1(weak_self, cloud))
        self.lidar2.listen(lambda cloud: self.set_lidar2(weak_self, cloud))
        self.gnss1.listen(lambda gnss: self.set_gnss1(weak_self, gnss))
        self.gnss2.listen(lambda gnss: self.set_gnss2(weak_self, gnss))


        calibration = np.identity(3)
        # calibration[0, 0] = 5
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        print("calibration = ")
        print(calibration)
        self.camera.calibration = calibration
        self.camera_A_rgb.calibration = calibration

    def control(self):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """
        # car = self.car_controlled
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True
        if keys[K_TAB]:
            self.car_controlled = self.car_A if self.car_controlled == self.car else self.car

        control = self.car_controlled.get_control()
        control.throttle = 0

        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        self.car_controlled.apply_control(control)
        return False

    @staticmethod
    def set_rgb_image1(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture_rgb1:
            self.image1 = img
            save_name = "./obstacle_detect/eval/image/000000.jpeg"
            cc = carla.ColorConverter.Raw
            # self.image1 = image.
            img.save_to_disk(save_name, cc)
            self.capture_rgb1 = False

    @staticmethod
    def set_rgb_image2(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture_rgb2:
            self.image2 = img
            self.capture_rgb2 = False

    @staticmethod
    def set_depth_image(weak_self, depth_img):
        """
        Sets depth image coming from depth camera sensor.
        The self.capture_depth flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        cc = carla.ColorConverter.Depth
        depth_img.save_to_disk('depth_img/00.png', cc)
        if self.capture_depth:
            distance = np.frombuffer(depth_img.raw_data, dtype=np.dtype("uint8"))
            distance = np.reshape(distance, (depth_img.height, depth_img.width, 4))
            distance = distance[:, :, :3]
            distance = distance.astype("uint32")
            distance = (distance[:, :, 2] + distance[:, :, 1] * 256 + distance[:, :, 0] * 256 * 256) / (
                        256 * 256 * 256 - 1) * 1000
            np.sum(self.mask, axis=0)
            distance[np.sum(self.mask, axis=0) == 0] = sys.maxunicode
            self.depth_image = distance
            # print('shape: ', np.shape(distance))
            self.capture_depth = False

        # plt.figure("depth")
        # plt.imshow(self.depth_image)

    @staticmethod
    def set_depth_image2(weak_self, depth_img):
        """
        Sets depth image coming from depth camera sensor of the back car.
        The self.capture_depth flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        cc = carla.ColorConverter.Depth
        depth_img.save_to_disk('depth_img/01.png', cc)
        if self.capture_depth2:
            self.capture_depth2 = False

    @staticmethod
    def set_lidar1(weak_self, cloud):
        self = weak_self()
        cloud.save_to_disk('_out/1%06d' % cloud.frame)
        if self.capture_lidar1:
            self.capture_lidar1 = False
            num_point = 0
            for location in cloud:
                num_point += 1
            self.cloud1 = np.zeros((num_point, 3))
            i = 0
            for location in cloud:
                self.cloud1[i, 0] = location.x
                self.cloud1[i, 1] = -location.y
                self.cloud1[i, 2] = -location.z + 0.7
                i += 1
            cloud_y_minus_z_x = np.zeros((num_point, 3))
            cloud_y_minus_z_x[:, 2] = self.cloud1[:, 1]
            cloud_y_minus_z_x[:, 1] = -self.cloud1[:, 2]
            cloud_y_minus_z_x[:, 0] = self.cloud1[:, 0]
            # print('shape2: ', np.shape(cloud_y_minus_z_x))
            distance = np.ones((VIEW_HEIGHT, VIEW_WIDTH))
            distance = distance * 50
            calibration = self.camera_A_rgb.calibration
            for point in cloud_y_minus_z_x:
                position = np.dot(calibration, point)
                position[0] = position[0]/position[2]
                position[1] = position[1]/position[2]
                # print(position)
                u = int(position[0])
                v = int(position[1])
                if (2 < position[2] < 50) and (0 <= v < VIEW_HEIGHT) and (0 <= u < VIEW_WIDTH) and (position[2] < distance[v, u]):
                    # if distance[int(v), int(u)] == 50 or distance[int(v), int(u)] < position[2]:
                    distance[v, u] = position[2]
            self.depth_image = distance
            # distance = distance * 25
            # distance = np.shape(vector, (VIEW_HEIGHT, VIEW_WIDTH))
            new_im = Image.fromarray(np.uint8(distance))
            # plt.imsave('depth_img/1.jpeg', new_im, cmap='gray')

    @staticmethod
    def set_lidar2(weak_self, cloud):
        self = weak_self()
        cloud.save_to_disk('_out/2%06d' % cloud.frame)
        if self.capture_lidar2:
            self.capture_lidar2 = False
            num_point = 0
            for location in cloud:
                num_point += 1
            # print(num_point)
            self.cloud2 = np.zeros((num_point, 3))
            i = 0
            for location in cloud:
                self.cloud2[i, 0] = location.x
                self.cloud2[i, 1] = -location.y
                self.cloud2[i, 2] = -location.z+0.7
                i += 1

    @staticmethod
    def set_gnss1(weak_self, gnss):
        self = weak_self()
        if self.capture_gnss1:
            # print(gnss)
            wgs84 = nv.FrameE(name='WGS84')
            point = wgs84.GeoPoint(gnss.latitude, gnss.longitude, gnss.altitude, degrees=True)
            # print(point)
            self.pointA = point
            self.capture_gnss1 = False

    @staticmethod
    def set_gnss2(weak_self, gnss):
        self = weak_self()
        if self.capture_gnss2:
            # print(gnss)
            wgs84 = nv.FrameE(name='WGS84')
            point = wgs84.GeoPoint(gnss.latitude, gnss.longitude, gnss.altitude, degrees=True)
            # print(point)
            self.point = point
            self.capture_gnss2 = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image2 is not None and self.image1 is not None:
            array1 = np.frombuffer(self.image1.raw_data, dtype=np.dtype("uint8"))
            array1 = np.reshape(array1, (self.image1.height, self.image1.width, 4))
            array1 = array1[:, :, :3]
            array1 = array1[:, :, ::-1].copy()

            array2 = np.frombuffer(self.image2.raw_data, dtype=np.dtype("uint8"))
            array2 = np.reshape(array2, (self.image2.height, self.image2.width, 4))
            array2 = array2[:, :, :3]
            array2 = array2[:, :, ::-1].copy()

            # print(array2)
            if self.view_uv_all is not None and self.mask is not None:
                indices = np.argsort(self.obstacle_distances)[::-1]
                for i in indices:
                    mask = self.mask[i]
                    valide_indices = np.nonzero(mask)
                    uv_couples = self.view_uv_all[i][valide_indices]
                    # print(np.shape(valide_indices))
                    array2[uv_couples[:, 1], uv_couples[:, 0]] = array1[valide_indices]
                    # array1[valide_indices] = [0, 0, 0]
                    # array2[valide_indices] = array1[valide_indices]

            array = np.concatenate((array1, array2), axis=1)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
        else:
            print("render not ok")

    def game_loop(self):
        """
        Main program loop.
        """

        saved = False
        # load weights

        weights_path = os.path.abspath(os.path.dirname(__file__)) + '/obstacle_detect/weights'
        model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
        if len(model_lst) == 0:
            print('No previous model found, please train first!')
            exit()
        else:
            print('Using previous model %s' % model_lst[-1])
            # TODO: load bins from file or something
            dir_weights = weights_path + '/%s' % model_lst[-1]
            fp = open(dir_weights, "rb")
            weights = torch.load(fp)

        # load yolo
        yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/obstacle_detect/weights'
        yolo = cv_Yolo(yolo_path)

        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            # self.setup_pedestrian()
            # self.setup_pedestrian_2()
            self.setup_car_0()
            self.setup_car_A()
            self.setup_car()
            self.setup_camera()

            self.car_controlled = self.car

            self.display = pygame.display.set_mode((VIEW_WIDTH * 2, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.1
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.car.set_autopilot(True)
            # vehicles = [vehicle for vehicle in self.world.get_actors().filter('vehicle.*')]
            # pedestrians = [pedestrian for pedestrian in self.world.get_actors().filter('walker.pedestrian.*')]



            # print(obstacles)

            while True:
                time = self.world.tick()

                '''
                figure1 = plt.figure()
                ax1 = Axes3D(figure1)
                ax1.scatter3D(self.cloud1[:, 0], self.cloud1[:, 1], self.cloud1[:, 2])
                plt.show()
                
                figure2 = plt.figure()
                ax2 = Axes3D(figure2)
                ax2.scatter3D(self.cloud2[:, 0], self.cloud2[:, 1], self.cloud2[:, 2])
                plt.show()
                time.sleep(5)
                print(np.shape(self.cloud1))
                print(np.shape(self.cloud2))
                
                plt.figure()
                plt.scatter(self.ay, self.ax)
                plt.show()
                time.sleep(5)
                '''
                self.obstacle_distances = []
                x1 = self.car_A.get_location().x
                x2 = self.car.get_location().x
                # print(x1, x2)
                bounding_boxes = detect(weights, yolo)
                # print(np.shape(bounding_boxes))
                if abs(x1 - x2) > 2:
                    T, R, i = icp.icp(self.cloud1, self.cloud2, max_iterations=1000, tolerance=0.001)

                    self.mask = ClientSideBoundingBox.create_bounding_area_masks(bounding_boxes, self.depth_image)

                    self.view_uv_all = ClientSideBoundingBox.get_2d_cords_sensor2(self.camera_A_rgb, self.depth_image,
                                                                                  self.mask, self.obstacle_distances,
                                                                                  self.camera_A_rgb, self.camera, T)

                self.render(self.display)

                ClientSideBoundingBox.draw_bounding_boxes(self.display, bounding_boxes)

                pygame.image.save(self.display, "screenshot/screenshot_" + str(time) + ".jpeg")
                pygame.display.flip()

                pygame.event.pump()
                self.capture_rgb1 = True
                self.capture_rgb2 = True
                self.capture_depth = True
                self.capture_depth2 = True
                self.detect_obstacle = True
                self.capture_lidar1 = True
                self.capture_lidar2 = True
                self.capture_gnss1 = True
                self.capture_gnss2 = True
                if self.control():
                    return


        finally:
            self.set_synchronous_mode(False)
            self.client.stop_recorder()
            self.camera.destroy()
            self.camera_A_depth.destroy()
            self.camera_A_rgb.destroy()
            self.car.destroy()
            self.car_A.destroy()
            self.pedestrian.destroy()
            # self.car_0.destroy()
            pygame.quit()

    def print_to_txt(self, data):
        if ~self.saved_file and (data is not None):
            np.savetxt('data.txt', data, fmt="%.3f")
            saved = True


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()