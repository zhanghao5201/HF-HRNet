# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import torchvision
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import math
import cv2

class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(self.color[i])

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(point_color[i])


# (R,G,B)
color3 = [(243,176,252),(243,176,252),(243,176,252),
    (240,176,0), (240,176,0), (240,176,0),
    (127,2,240),(127,2,240),(255,0,0),
    (0,255,255), (0,255,255),(0,255,255),
    (142, 209, 169),(142, 209, 169),(142, 209, 169)]

link_pairs3 = [
        [5, 4], [4, 3], [3,6],[6,2],[2,1],[1,0],
        [6, 7], [7, 8], [8, 9],
        [7, 13],[13, 14],[14, 15],
        [7,12], [12,11], [11,10]
        ]
point_color3 = [(240,176,0),(240,176,0),(240,176,0),
            (243,176,252), (243,176,252),(243,176,252),
            (127,2,240),(127,2,240),(255,0,0),(255,0,0),
            (142, 209, 169),(142, 209, 169),(142, 209, 169),
            (0,255,255),(0,255,255),(0,255,255)]


zhanghao_style = ColorStyle(color3, link_pairs3, point_color3)


def save_debug_images(input, meta, joints_pred,  prefix):
    
    colorstyle =zhanghao_style

    save_batch_image_test(input, joints_pred,meta['joints_vis'], '{}_pred1.jpg'.format(prefix),colorstyle.link_pairs,colorstyle.ring_color)

def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict
def save_batch_image_test(batch_image, batch_joints, batch_joints_vis, file_name, link_pairs,ring_color):
    '''
    #batch_image: [batch_size, channel, height, width]
    #batch_joints: [batch_size, num_joints, 3],
    #batch_joints_vis: [batch_size, num_joints, 1],
    '''
    grid = torchvision.utils.make_grid(batch_image, 1, 0, True)#画图，nrow每行显式的图片数
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()#交换维度
    ndarr = ndarr.copy()
    #ndarr=cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    nmaps = batch_image.size(0)
    xmaps = min(1, nmaps)#一行几张图片
    ymaps = int(math.ceil(float(nmaps) / xmaps))#有几行
    height = int(batch_image.size(2) +0)
    width = int(batch_image.size(3) + 0)
    k = 0
    h=0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break#跳出for循环
            joints = batch_joints[k]
            joints_dict = map_joint_dict(joints)
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + 0 + joint[0]
                joint[1] = y * height + 0 + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, ring_color[h], 2)
                    #cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [0,255,0], 2)
                h=h+1
                if h==16:
                    h=0
            for i, link_pair in enumerate(link_pairs):
                lw = cv2.LINE_4
                if joints_vis[link_pair[0]][0] and joints_vis[link_pair[1]][0] :
                    cv2.line(ndarr,(joints_dict[link_pair[0]][0],joints_dict[link_pair[0]][1]),(joints_dict[link_pair[1]][0],joints_dict[link_pair[1]][1]),link_pair[2],3,lw)
            k=k+1
    #cv2.imwrite(file_name, ndarr)

def single_gpu_test(model, data_loader):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.


    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    #save_debug_images(input, meta, joints_pred,  prefix):
    #print(len(results),results[0])
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        if rank == 0:
            # use the first key as main key to calculate the batch size
            batch_size = len(next(iter(data.values())))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    print("kndsnf")
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results in cpu mode.

    It saves the results on different gpus to 'tmpdir' and collects
    them by the rank 0 worker.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None

    Returns:
        list: Ordered results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # synchronizes all processes to make sure tmpdir exist
    dist.barrier()
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    # synchronizes all processes for loading pickle file
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None

    # load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        part_list.append(mmcv.load(part_file))
    # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    # remove tmp dir
    shutil.rmtree(tmpdir)
    return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results in gpu mode.

    It encodes results to gpu tensors and use gpu communication for results
    collection.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.

    Returns:
        list: Ordered results.
    """

    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    return None
