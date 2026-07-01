import cv2
import os
import sys
import torch
import torchvision.utils
import numpy as np
import yaml
from collections import OrderedDict

# ==============================
# IMPORT LIVEPORTRAIT
# ==============================
_lp_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liveportrait_src", "src")
sys.path.insert(0, _lp_src_dir)

from modules.appearance_feature_extractor import AppearanceFeatureExtractor
from modules.motion_extractor import MotionExtractor
from modules.warping_network import WarpingNetwork
from modules.spade_generator import SPADEDecoder
from modules.stitching_retargeting_network import StitchingRetargetingNetwork
from utils.camera import headpose_pred_to_degree, get_rotation_matrix


# ==============================
# UTIL
# ==============================
def _remove_ddp_key(state_dict):
    new_sd = OrderedDict()
    for key in state_dict.keys():
        new_sd[key.replace("module.", "")] = state_dict[key]
    return new_sd


def load_model(ckpt_path, model_config, device, model_type):
    model_params = model_config["model_params"][f"{model_type}_params"]

    if model_type == "appearance_feature_extractor":
        model = AppearanceFeatureExtractor(**model_params).to(device)

    elif model_type == "motion_extractor":
        model = MotionExtractor(**model_params).to(device)

    elif model_type == "warping_module":
        model = WarpingNetwork(**model_params).to(device)

    elif model_type == "spade_generator":
        model = SPADEDecoder(**model_params).to(device)

    elif model_type == "stitching_retargeting_module":
        config = model_config["model_params"]["stitching_retargeting_module_params"]
        checkpoint = torch.load(ckpt_path, map_location=device)

        stitcher = StitchingRetargetingNetwork(**config.get("stitching"))
        stitcher.load_state_dict(_remove_ddp_key(checkpoint["retarget_shoulder"]))
        stitcher = stitcher.to(device).eval()

        retargetor_lip = StitchingRetargetingNetwork(**config.get("lip"))
        retargetor_lip.load_state_dict(_remove_ddp_key(checkpoint["retarget_mouth"]))
        retargetor_lip = retargetor_lip.to(device).eval()

        retargetor_eye = StitchingRetargetingNetwork(**config.get("eye"))
        retargetor_eye.load_state_dict(_remove_ddp_key(checkpoint["retarget_eye"]))
        retargetor_eye = retargetor_eye.to(device).eval()

        return {
            "stitching": stitcher,
            "lip": retargetor_lip,
            "eye": retargetor_eye
        }

    else:
        raise ValueError("Unknown model type")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


# ==============================
# MAIN CLASS
# ==============================
class PortraitAnimator:

    def __init__(self, weights_dir="./weights", device="cuda"):
        print("[AI] Initialisation...")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_dir = weights_dir

        config_path = os.path.join(_lp_src_dir, "config", "models.yaml")
        with open(config_path, "r") as f:
            model_config = yaml.safe_load(f)

        self.appearance_extractor = load_model(
            os.path.join(weights_dir, "appearance_feature_extractor.pth"),
            model_config, self.device, "appearance_feature_extractor"
        )

        self.motion_extractor = load_model(
            os.path.join(weights_dir, "motion_extractor.pth"),
            model_config, self.device, "motion_extractor"
        )

        self.warping_module = load_model(
            os.path.join(weights_dir, "warping_module.pth"),
            model_config, self.device, "warping_module"
        )

        self.spade_generator = load_model(
            os.path.join(weights_dir, "spade_generator.pth"),
            model_config, self.device, "spade_generator"
        )

        self.stitching_module = load_model(
            os.path.join(weights_dir, "stitching_retargeting_module.pth"),
            model_config, self.device, "stitching_retargeting_module"
        )

        self.reset_cache()

        print("[AI] Ready.")

    def reset_cache(self):
        self.source_feature = None
        self.source_kp_info = None
        self.source_kp = None
        self.initial_driving_kp_info = None
        self.initial_driving_kp = None

    # ==============================
    # KEYPOINT EXTRACTION
    # ==============================
    def get_kp_info(self, x):
        kp_info = self.motion_extractor(x)

        for k, v in kp_info.items():
            if isinstance(v, torch.Tensor):
                kp_info[k] = v.float()

        bs = kp_info["kp"].shape[0]

        kp_info["pitch"] = headpose_pred_to_degree(kp_info["pitch"])[:, None]
        kp_info["yaw"] = headpose_pred_to_degree(kp_info["yaw"])[:, None]
        kp_info["roll"] = headpose_pred_to_degree(kp_info["roll"])[:, None]

        kp_info["kp"] = kp_info["kp"].reshape(bs, -1, 3)
        kp_info["exp"] = kp_info["exp"].reshape(bs, -1, 3)

        return kp_info

    def transform_keypoint(self, kp_info):
        kp = kp_info["kp"]
        exp = kp_info["exp"]
        t = kp_info["t"]
        scale = kp_info["scale"]

        rot = get_rotation_matrix(kp_info["pitch"], kp_info["yaw"], kp_info["roll"])

        kp = kp @ rot + exp
        kp *= scale[..., None]
        kp[:, :, :2] += t[:, None, :2]

        return kp

    # ==============================
    # MAIN INFERENCE
    # ==============================
    @torch.inference_mode()
    def generate_frame(self, source_image, driving_frame, narration_mode=False):

        source = self.preprocess(source_image).to(self.device)
        driving = self.preprocess(driving_frame).to(self.device)

        if self.source_feature is None:
            self.source_feature = self.appearance_extractor(source)
            self.source_kp_info = self.get_kp_info(source)
            self.source_kp = self.transform_keypoint(self.source_kp_info)

        driving_kp_info = self.get_kp_info(driving)
        driving_kp = self.transform_keypoint(driving_kp_info)

        if self.initial_driving_kp_info is None:
            self.initial_driving_kp_info = {k: v.clone() for k, v in driving_kp_info.items() if isinstance(v, torch.Tensor)}
            self.initial_driving_kp = driving_kp.clone()

        # ==========================
        # 🔥 BOOST EXPRESSIONS
        # ==========================

        if narration_mode:
            MOTION = 0.00
            EXPRESSION = 0.65
            STABILIZER = 0.55
        else:
            MOTION = 0.30
            EXPRESSION = 0.50
            STABILIZER = 0.85

        delta = driving_kp - self.initial_driving_kp
        exp_delta = driving_kp_info["exp"] - self.initial_driving_kp_info["exp"]

        kp_relative = self.source_kp + (delta * MOTION + exp_delta * EXPRESSION) * STABILIZER

        # ==========================

        out = self.warping_module(
            self.source_feature,
            kp_driving=kp_relative,
            kp_source=self.source_kp
        )

        out = self.spade_generator(feature=out["out"])

        return self.tensor_to_bgr(out)

    # ==============================
    # UTILS
    # ==============================
    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    def tensor_to_bgr(self, tensor):
        img = tensor.squeeze(0).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))

        if img.min() < 0:
            img = (img + 1) / 2

        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)