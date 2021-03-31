import torch
import torch.nn.functional as F
# from models.utils import SpatialTransformer
import time
def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid


def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid

def disp_warp(right_input, disparity_samples, padding_mode='border'):
    device = right_input.get_device()
    left_y_coordinate = torch.arange(0.0, right_input.size()[3], device=device).repeat(right_input.size()[2])
    left_y_coordinate = left_y_coordinate.view(right_input.size()[2], right_input.size()[3])
    left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=right_input.size()[3] - 1)
    left_y_coordinate = left_y_coordinate.expand(right_input.size()[0], -1, -1) #[B,H,W]

    right_feature_map = right_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
    disparity_samples = disparity_samples.float()
    #[B,C,H,W] - [B,C,H,W]
    right_y_coordinate = left_y_coordinate.expand(
        disparity_samples.size()[1], -1, -1, -1).permute([1, 0, 2, 3]) - disparity_samples

    right_y_coordinate_1 = right_y_coordinate
    right_y_coordinate = torch.clamp(right_y_coordinate, min=0, max=right_input.size()[3] - 1)
    # torch.cuda.synchronize()
    # start  = time.time()
    warped_right_feature_map = torch.gather(right_feature_map, dim=4, index=right_y_coordinate.expand(
        right_input.size()[1], -1, -1, -1, -1).permute([1, 0, 2, 3, 4]).long())  #索引坐标取整了
    # torch.cuda.synchronize()
    # temp_time = time.time() - start
    # print('gather_time = {:3f}'.format(temp_time))
    right_y_coordinate_1 = right_y_coordinate_1.unsqueeze(1)
    warped_right_feature_map = (1 - ((right_y_coordinate_1 < 0) +
                                     (right_y_coordinate_1 > right_input.size()[3] - 1)).float()) * \
        (warped_right_feature_map) + torch.zeros_like(warped_right_feature_map)
    #去除视差维度
    return warped_right_feature_map.squeeze(2)
# def disp_warp(img, disp, padding_mode='border'):
    # """Warping by disparity
    # Args:
        # img: [B, 3, H, W]
        # disp: [B, 1, H, W], positive
        # padding_mode: 'zeros' or 'border'
    # Returns:
        # warped_img: [B, 3, H, W]
        # valid_mask: [B, 3, H, W]
    # """
    # assert disp.min() >= 0

    # grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # # Note that -disp here
    # offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    # sample_grid = grid + offset
    # sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    # warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode)
    # #使用类似STN方法
    # mask = torch.ones_like(img)
    # valid_mask = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='zeros')
    # valid_mask[valid_mask < 0.9999] = 0
    # valid_mask[valid_mask > 0] = 1
    # return warped_img, valid_mask
