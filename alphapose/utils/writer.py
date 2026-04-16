import os
import time
from threading import Thread
from queue import Queue

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms, write_json

DEFAULT_VIDEO_SAVE_OPT = {
    'savepath': 'examples/res/1.mp4',
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (640, 480)
}

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
COCO_LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (17, 11), (17, 12), (11, 13), (12, 14), (13, 15), (14, 16)
]
HALPE26_LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),
    (17, 18), (18, 19), (19, 11), (19, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
    (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25)
]


class DataWriter():
    def __init__(self, cfg, opt, save_video=False,
                 video_save_opt=DEFAULT_VIDEO_SAVE_OPT,
                 queueSize=1024):
        self.cfg = cfg
        self.opt = opt
        self.video_save_opt = video_save_opt

        self.eval_joints = EVAL_JOINTS
        self.save_video = save_video
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)

        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')
        self.save_mask = bool(getattr(opt, 'save_mask', False))
        self.save_mask_vis = bool(getattr(opt, 'save_mask_vis', False))
        if self.save_mask:
            os.makedirs(os.path.join(opt.outputpath, 'masks'), exist_ok=True)
        if self.save_mask_vis:
            os.makedirs(os.path.join(opt.outputpath, 'masks_vis'), exist_ok=True)

        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper
            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, 'poseflow'))

        if self.opt.save_img or self.save_video or self.opt.vis:
            loss_type = self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss')
            num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
            if loss_type == 'MSELoss':
                self.vis_thres = [0.4] * num_joints
            elif 'JointRegression' in loss_type:
                self.vis_thres = [0.05] * num_joints
            elif loss_type == 'Combined':
                if num_joints == 68:
                    hand_face_num = 42
                else:
                    hand_face_num = 110
                self.vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num

        self.use_heatmap_loss = (self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss')

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self

    def update(self):
        final_result = []
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        if self.save_video:
            # initialize the file video stream, adapt ouput video resolution to original video
            stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            if not stream.isOpened():
                print("Try to use other video encoders...")
                ext = self.video_save_opt['savepath'].split('.')[-1]
                fourcc, _ext = self.recognize_video_ext(ext)
                self.video_save_opt['fourcc'] = fourcc
                self.video_save_opt['savepath'] = self.video_save_opt['savepath'][:-4] + _ext
                stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            assert stream.isOpened(), 'Cannot open video for writing'
        # keep looping infinitelyd
        while True:
            # ensure the queue is not empty and get item
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.wait_and_get(self.result_queue)
            if orig_img is None:
                # if the thread indicator variable is set (img is None), stop the thread
                if self.save_video:
                    stream.release()
                outputfile = getattr(self.opt, 'outputfile', 'alphapose-results.json')
                write_json(
                    final_result,
                    self.opt.outputpath,
                    form=self.opt.format,
                    for_eval=self.opt.eval,
                    outputfile=outputfile
                )
                print("Results have been written to json.")
                return
            # image channel RGB->BGR
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
            if boxes is None or len(boxes) == 0:
                if self.opt.save_img or self.save_video or self.opt.vis:
                    self.write_image(orig_img, im_name, stream=stream if self.save_video else None)
            else:
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                assert hm_data.dim() == 4

                face_hand_num = 110
                if hm_data.size()[1] == 136:
                    self.eval_joints = [*range(0,136)]
                elif hm_data.size()[1] == 26:
                    self.eval_joints = [*range(0,26)]
                elif hm_data.size()[1] == 133:
                    self.eval_joints = [*range(0,133)]
                elif hm_data.size()[1] == 68:
                    face_hand_num = 42
                    self.eval_joints = [*range(0,68)]
                elif hm_data.size()[1] == 21:
                    self.eval_joints = [*range(0,21)]
                pose_coords = []
                pose_scores = []
                for i in range(hm_data.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    if isinstance(self.heatmap_to_coord, list):
                        pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                            hm_data[i][self.eval_joints[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type)
                        pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                            hm_data[i][self.eval_joints[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                        pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                        pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
                    else:
                        pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)
                if not self.opt.pose_track:
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                        pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area, use_heatmap_loss=self.use_heatmap_loss)

                _result = []
                for k in range(len(scores)):
                    _result.append(
                        {
                            'keypoints':preds_img[k],
                            'kp_score':preds_scores[k],
                            'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                            'idx':ids[k],
                            'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                        }
                    )

                result = {
                    'imgname': im_name,
                    'result': _result
                }


                if self.opt.pose_flow:
                    poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                    for i in range(len(poseflow_result)):
                        result['result'][i]['idx'] = poseflow_result[i]['idx']

                final_result.append(result)
                if self.save_mask or self.save_mask_vis:
                    self.write_masks(orig_img, im_name, result)
                if self.opt.save_img or self.save_video or self.opt.vis:
                    if hm_data.size()[1] == 49:
                        from alphapose.utils.vis import vis_frame_dense as vis_frame
                    elif self.opt.vis_fast:
                        from alphapose.utils.vis import vis_frame_fast as vis_frame
                    else:
                        from alphapose.utils.vis import vis_frame
                    img = vis_frame(orig_img, result, self.opt, self.vis_thres)
                    self.write_image(img, im_name, stream=stream if self.save_video else None)

    def write_image(self, img, im_name, stream=None):
        if self.opt.vis:
            cv2.imshow("AlphaPose Demo", img)
            cv2.waitKey(30)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, 'vis', im_name), img)
        if self.save_video:
            stream.write(img)

    def _as_int_id(self, idx_value, fallback):
        if isinstance(idx_value, list) and len(idx_value) > 0:
            idx_value = sorted(idx_value)[0]
        try:
            if torch.is_tensor(idx_value):
                return int(idx_value.item())
            return int(float(idx_value))
        except Exception:
            return int(fallback)

    def _get_limb_pairs(self, kp_num):
        if kp_num == 26:
            return HALPE26_LIMBS
        if kp_num == 17:
            return COCO_LIMBS
        return []

    def _person_mask_from_pose(self, kp_preds, kp_scores, bbox, img_h, img_w):
        kp_num = kp_preds.shape[0]
        limb_pairs = self._get_limb_pairs(kp_num)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        conf_thres = 0.12
        x0, y0, x1, y1 = [float(v) for v in bbox]
        person_scale = max(10.0, max(x1 - x0, y1 - y0))
        joint_radius = max(2, int(round(person_scale * 0.015)))
        limb_thickness = max(3, int(round(person_scale * 0.05)))

        valid_points = []
        for j in range(kp_num):
            score = float(kp_scores[j])
            if score < conf_thres:
                continue
            x = int(round(float(kp_preds[j, 0])))
            y = int(round(float(kp_preds[j, 1])))
            if x < 0 or x >= img_w or y < 0 or y >= img_h:
                continue
            valid_points.append((x, y))
            cv2.circle(mask, (x, y), joint_radius, 255, thickness=-1)

        for (a, b) in limb_pairs:
            if a >= kp_num or b >= kp_num:
                continue
            if float(kp_scores[a]) < conf_thres or float(kp_scores[b]) < conf_thres:
                continue
            p1 = (int(round(float(kp_preds[a, 0]))), int(round(float(kp_preds[a, 1]))))
            p2 = (int(round(float(kp_preds[b, 0]))), int(round(float(kp_preds[b, 1]))))
            cv2.line(mask, p1, p2, 255, thickness=limb_thickness)

        # Add an explicit head circle so the mask covers the whole head region.
        # COCO/HALPE share head landmarks in indices 0..4 (nose/eyes/ears).
        head_ids = [0, 1, 2, 3, 4]
        head_points = []
        for head_idx in head_ids:
            if head_idx >= kp_num:
                continue
            if float(kp_scores[head_idx]) < conf_thres:
                continue
            hx = float(kp_preds[head_idx, 0])
            hy = float(kp_preds[head_idx, 1])
            if hx < 0 or hx >= img_w or hy < 0 or hy >= img_h:
                continue
            head_points.append((hx, hy))
        if head_points:
            head_arr = np.array(head_points, dtype=np.float32)
            head_center = np.mean(head_arr, axis=0)
            if head_arr.shape[0] >= 2:
                # Spread of visible head points gives a data-driven head radius.
                max_dist = float(np.max(np.linalg.norm(head_arr - head_center, axis=1)))
            else:
                max_dist = 0.0
            head_radius = int(round(max(8.0, max_dist * 1.8, person_scale * 0.085)))
            cv2.circle(
                mask,
                (int(round(head_center[0])), int(round(head_center[1]))),
                head_radius,
                255,
                thickness=-1
            )

        if len(valid_points) >= 3:
            hull = cv2.convexHull(np.array(valid_points, dtype=np.int32).reshape(-1, 1, 2))
            cv2.fillConvexPoly(mask, hull, 255)

        kernel_size = max(3, int(round(person_scale * 0.04)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def write_masks(self, orig_img, im_name, result):
        img_h, img_w = orig_img.shape[:2]
        stem = os.path.splitext(os.path.basename(im_name))[0]
        overlay = orig_img.copy()

        for person_idx, human in enumerate(result['result']):
            kp_preds = human['keypoints'].detach().cpu().numpy()
            kp_scores = human['kp_score'].detach().cpu().numpy().reshape(-1)
            box = human.get('box', [0, 0, img_w, img_h])
            bbox_xyxy = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            person_mask = self._person_mask_from_pose(kp_preds, kp_scores, bbox_xyxy, img_h, img_w)
            person_label = f'person-{person_idx}'
            tracking_id = self._as_int_id(human.get('idx', person_idx), person_idx)

            if self.save_mask:
                mask_name = f'{stem}_{person_label}.png'
                cv2.imwrite(os.path.join(self.opt.outputpath, 'masks', mask_name), person_mask)

            if self.save_mask_vis:
                color = (0, 180, 255) if person_idx % 2 == 0 else (255, 120, 0)
                color_layer = np.zeros_like(overlay)
                color_layer[:, :] = color
                person_region = person_mask > 0
                overlay[person_region] = cv2.addWeighted(
                    overlay[person_region], 0.35, color_layer[person_region], 0.65, 0
                )
                ys, xs = np.where(person_region)
                if ys.size > 0 and xs.size > 0:
                    cv2.putText(
                        overlay,
                        f'{person_label} (id {tracking_id})',
                        (int(xs.min()), max(12, int(ys.min()) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1
                    )

        if self.save_mask_vis:
            cv2.imwrite(os.path.join(self.opt.outputpath, 'masks_vis', os.path.basename(im_name)), overlay)

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)
        self.result_worker.join()

    def terminate(self):
        # directly terminate
        self.result_worker.terminate()

    def clear_queues(self):
        self.clear(self.result_queue)
        
    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def results(self):
        # return final result
        print(self.final_result)
        return self.final_result

    def recognize_video_ext(self, ext=''):
        if ext == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
        elif ext == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        elif ext == 'mov':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'
