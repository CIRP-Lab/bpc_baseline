import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R  # For metric evaluation

#############################################
# Differentiable Helper Functions (PyTorch)
#############################################

def geodesic_distance_from_matrix(R1, R2):
    """
    Compute the geodesic distance (angle in radians) between two rotation matrices.
    R1, R2: (B, 3, 3)
    """
    R_rel = torch.bmm(R1.transpose(1, 2), R2)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    arg = (trace - 1) / 2
    arg = torch.clamp(arg, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(arg)
    return theta

def rotmat_from_euler(euler_angles):
    """
    Convert Euler angles (B,3) to rotation matrices (B,3,3) in a differentiable manner.
    Assumes intrinsic 'xyz' order and angles in radians.
    """
    cx = torch.cos(euler_angles[:, 0])
    sx = torch.sin(euler_angles[:, 0])
    cy = torch.cos(euler_angles[:, 1])
    sy = torch.sin(euler_angles[:, 1])
    cz = torch.cos(euler_angles[:, 2])
    sz = torch.sin(euler_angles[:, 2])
    
    R11 = cy * cz
    R12 = -cy * sz
    R13 = sy
    R21 = sx * sy * cz + cx * sz
    R22 = -sx * sy * sz + cx * cz
    R23 = -sx * cy
    R31 = -cx * sy * cz + sx * sz
    R32 = cx * sy * sz + sx * cz
    R33 = cx * cy
    
    R_mat = torch.stack([
        torch.stack([R11, R12, R13], dim=1),
        torch.stack([R21, R22, R23], dim=1),
        torch.stack([R31, R32, R33], dim=1)
    ], dim=1)
    return R_mat

def quat_to_rotmat(quat):
    """
    Convert quaternions (B,4) to rotation matrices (B,3,3).
    Assumes quaternion ordering [x, y, z, w].
    """
    quat = quat / quat.norm(dim=1, keepdim=True)
    x, y, z, w = quat.unbind(dim=1)
    B = quat.shape[0]
    R_mat = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
        2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y,
    ], dim=1).view(B, 3, 3)
    return R_mat

def rotmat_from_6d(rep6d):
    """
    Convert 6D rotation representation to rotation matrices.
    Based on Zhou et al. (CVPR 2019).
    rep6d: (B,6) interpreted as two 3D vectors.
    """
    B = rep6d.shape[0]
    a1 = rep6d[:, 0:3]
    a2 = rep6d[:, 3:6]
    b1 = F.normalize(a1, dim=1, eps=1e-8)
    tmp = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(tmp, dim=1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=1)
    R_mat = torch.stack([b1, b2, b3], dim=2)
    return R_mat

def euler_to_quat_torch(euler):
    """
    Convert a batch of Euler angles (B,3) to quaternions (B,4) using intrinsic 'xyz' order.
    Returns quaternions with ordering [x, y, z, w].
    """
    half = euler * 0.5
    cx = torch.cos(half[:, 0])
    sx = torch.sin(half[:, 0])
    cy = torch.cos(half[:, 1])
    sy = torch.sin(half[:, 1])
    cz = torch.cos(half[:, 2])
    sz = torch.sin(half[:, 2])
    
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    
    quat = torch.stack([x, y, z, w], dim=1)
    return quat

#############################################
# Helper Functions for Metric Evaluation (using SciPy)
#############################################

def compute_rot_deg_mean_matrix(R_gt, R_pred):
    """
    Compute the mean angular error (in degrees) between batches of rotation matrices.
    Both R_gt and R_pred are torch tensors of shape (B, 3, 3).
    """
    R_gt_np = R_gt.detach().cpu().numpy()
    R_pred_np = R_pred.detach().cpu().numpy()
    r1 = R.from_matrix(R_gt_np)
    r2 = R.from_matrix(R_pred_np)
    r_relative = r1.inv() * r2
    angles = r_relative.magnitude()
    return torch.tensor(np.mean(np.degrees(angles)), dtype=torch.float32)

#############################################
# Loss Classes
#############################################

class EulerAnglePoseLoss(nn.Module):
    """
    Loss based on Euler angles.
    Uses the groundtruth rotation matrix if available.
    """
    def __init__(self, w_rot=1.0):
        super(EulerAnglePoseLoss, self).__init__()
        self.w_rot = w_rot

    def forward(self, labels, preds):
        # Use GT rotation matrix if available.
        if "R" in labels:
            R_gt = labels["R"]
        else:
            gt_euler = labels["euler"]
            R_gt = rotmat_from_euler(gt_euler)
        pred_euler = preds[:, :3]
        R_pred = rotmat_from_euler(torch.remainder(pred_euler + math.pi, 2 * math.pi) - math.pi)
        angles = geodesic_distance_from_matrix(R_pred, R_gt)
        rotation_loss = angles.mean()
        loss_val = self.w_rot * rotation_loss
        rot_deg_mean = compute_rot_deg_mean_matrix(R_gt, R_pred)
        metrics = {
            "rot_loss": loss_val,
            "rot_deg_mean": rot_deg_mean,
        }
        return loss_val, metrics

class QuaternionPoseLoss(nn.Module):
    """
    Loss using quaternion representation.
    Uses the GT rotation matrix if available.
    """
    def __init__(self, w_rot=1.0):
        super(QuaternionPoseLoss, self).__init__()
        self.w_rot = w_rot

    def forward(self, labels, preds):
        if "R" in labels:
            R_gt = labels["R"]
        else:
            gt_euler = labels["euler"]
            gt_quat = euler_to_quat_torch(gt_euler)
            R_gt = quat_to_rotmat(gt_quat)
        pred_quat = preds[:, :4]
        eps = 1e-8
        pred_quat = pred_quat / (pred_quat.norm(dim=1, keepdim=True) + eps)
        # Ensure consistency in sign.
        dot = (pred_quat * pred_quat).sum(dim=1, keepdim=True)
        pred_quat = torch.where(dot < 0, -pred_quat, pred_quat)
        R_pred = quat_to_rotmat(pred_quat)
        angles = geodesic_distance_from_matrix(R_pred, R_gt)
        rotation_loss = angles.mean()
        loss_val = self.w_rot * rotation_loss
        rot_deg_mean = compute_rot_deg_mean_matrix(R_gt, R_pred)
        metrics = {
            "rot_loss": loss_val,
            "rot_deg_mean": rot_deg_mean,
        }
        return loss_val, metrics

class SixDPoseLoss(nn.Module):
    """
    Loss using 6D rotation representation.
    Uses the GT rotation matrix if available.
    """
    def __init__(self, w_rot=1.0):
        super(SixDPoseLoss, self).__init__()
        self.w_rot = w_rot

    def forward(self, labels, preds):
        if "R" in labels:
            R_gt = labels["R"]
        else:
            gt_euler = labels["euler"]
            R_gt = rotmat_from_euler(gt_euler)
        rep6d = preds[:, :6]
        R_pred = rotmat_from_6d(rep6d)
        angles = geodesic_distance_from_matrix(R_pred, R_gt)
        rotation_loss = angles.mean()
        loss_val = self.w_rot * rotation_loss
        rot_deg_mean = compute_rot_deg_mean_matrix(R_gt, R_pred)
        metrics = {
            "rot_loss": loss_val,
            "rot_deg_mean": rot_deg_mean,
        }
        return loss_val, metrics
