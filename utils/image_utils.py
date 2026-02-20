"""
工具函数：将起点、终点、障碍物渲染成单张图像
"""
import numpy as np
import torch


def render_scene_to_image(start, goal, obstacle_map, image_size=(64, 64), map_size=8):
    """
    将起点、终点和障碍物地图渲染成一张多通道图像
    
    Args:
        start: (2,) 起点坐标，范围 [0, map_size]
        goal: (2,) 终点坐标，范围 [0, map_size]
        obstacle_map: (H, W) 障碍物二值地图
        image_size: 输出图像尺寸
        map_size: 地图坐标范围
        
    Returns:
        image: (3, H, W) 三通道图像
            - channel 0: 障碍物层
            - channel 1: 起点层
            - channel 2: 终点层
    """
    H, W = image_size
    image = np.zeros((3, H, W), dtype=np.float32)
    
    # Channel 0: 障碍物 (上采样到目标尺寸)
    if obstacle_map.ndim == 2:
        from scipy.ndimage import zoom
        scale_h = H / obstacle_map.shape[0]
        scale_w = W / obstacle_map.shape[1]
        image[0] = zoom(obstacle_map, (scale_h, scale_w), order=0)  # 最近邻插值
    
    # Channel 1: 起点 (高斯blob)
    start_pixel = (start / map_size * np.array([H, W])).astype(int)
    start_pixel = np.clip(start_pixel, 0, [H-1, W-1])
    image[1] = _create_gaussian_blob(H, W, start_pixel, sigma=2.0)
    
    # Channel 2: 终点 (高斯blob)
    goal_pixel = (goal / map_size * np.array([H, W])).astype(int)
    goal_pixel = np.clip(goal_pixel, 0, [H-1, W-1])
    image[2] = _create_gaussian_blob(H, W, goal_pixel, sigma=2.0)
    
    return image


def _create_gaussian_blob(H, W, center, sigma=2.0):
    """在指定位置创建高斯blob"""
    y, x = np.ogrid[0:H, 0:W]
    cy, cx = center
    dist_sq = (y - cy)**2 + (x - cx)**2
    blob = np.exp(-dist_sq / (2 * sigma**2))
    return blob


def render_scene_single_channel(start, goal, obstacle_map, image_size=(64, 64), map_size=8):
    """
    渲染成单通道图像，不同元素用不同值表示
    
    Returns:
        image: (1, H, W) 单通道图像
            - 0: 空白
            - 0.5: 障碍物
            - 0.8: 起点
            - 1.0: 终点
    """
    H, W = image_size
    image = np.zeros((H, W), dtype=np.float32)
    
    # 障碍物
    if obstacle_map.ndim == 2:
        from scipy.ndimage import zoom
        scale_h = H / obstacle_map.shape[0]
        scale_w = W / obstacle_map.shape[1]
        obs_resized = zoom(obstacle_map, (scale_h, scale_w), order=0)
        image[obs_resized > 0] = 0.5
    
    # 起点和终点
    start_pixel = (start / map_size * np.array([H, W])).astype(int)
    start_pixel = np.clip(start_pixel, 0, [H-1, W-1])
    goal_pixel = (goal / map_size * np.array([H, W])).astype(int)
    goal_pixel = np.clip(goal_pixel, 0, [H-1, W-1])
    
    # 画十字标记
    _draw_cross(image, start_pixel, value=0.8, size=2)
    _draw_cross(image, goal_pixel, value=1.0, size=2)
    
    return image[np.newaxis, ...]  # (1, H, W)


def _draw_cross(image, center, value=1.0, size=2):
    """在图像上画十字"""
    cy, cx = center
    H, W = image.shape
    for i in range(-size, size+1):
        if 0 <= cy+i < H:
            image[cy+i, cx] = value
        if 0 <= cx+i < W:
            image[cy, cx+i] = value
