# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

threshold = 2.21

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()  # deque 是一个双端队列

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))
    
    def run_on_video_flow(self, video, output_name):
        """
        Visualizes predictions on frames of the input video with Optical Flow Method.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame
        
        def flow_visualization(prev, flow):
            # 绘制线
            step=10
            gray = prev
            h, w = gray.shape[:2]
            y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T
            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines)

            line = []
            for l in lines:
                if l[0][0]-l[1][0]>3 or l[0][1]-l[1][1]>3:
                    # print(l)
                    line.append(l)

            cv2.polylines(frame, line, 0, (0,255,255))
            cv2.imwrite('flow.jpg', frame)
        
        def should_send(prev, pred, flow):
            pre_box = pred['instances'].get('pred_boxes')
            if len(pre_box) == 0:
                return 1, -1, -1
            pre_mask = pred['instances'].get('pred_masks')
            
            boxes = pre_box.tensor

            step = 10
            h, w = prev.shape[:2]
            y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T
            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines)
            # print(lines)

            def movement(boxes, lines):
                total_move = np.zeros([boxes.shape[0], 4])
                num = np.zeros([boxes.shape[0], 4])
                xylist = []
                for i in range(boxes.shape[0]):
                    xylist.append(boxes[i].tolist())
                for l in lines:
                    for i in range(len(xylist)):
                        if l[0][0] > xylist[i][0] and l[0][0] < xylist[i][2] and l[0][1] > xylist[i][1] and l[0][1] < xylist[i][3]:
                            total_move[i][0] += l[0][0]-l[1][0]
                            total_move[i][2] += l[0][0]-l[1][0]
                            total_move[i][1] += l[0][1]-l[1][1]
                            total_move[i][3] += l[0][1]-l[1][1]
                            num[i] += 1
                move = np.true_divide(total_move, num)
                # print(move)
                return total_move, move
            
            total_move, move = movement(boxes, lines)
            total_area = pred['instances'].get('pred_boxes').area().cpu().numpy()
            avg_mov = np.sum(np.true_divide(np.sum(total_move, axis=1), total_area))

            if abs(avg_mov)*10 > threshold:
                should = 1
            else:
                should = 0
                new_tensor = pre_box.tensor.cpu() + move
                new_box = Boxes(new_tensor)
                new_mask = pre_mask.cpu().numpy()
                # print(new_mask)
                row, col = new_mask[0].shape
                for i in range(new_mask.shape[0]):
                    origin = new_mask[i]
                    if i == 0 or i == 1:
                        addr = "mask" + str(i) + ".txt"
                        np.savetxt(addr, origin)
                    if not np.isnan(move[i][0]) and move[i][0] != 0:
                        # print(move[i][0])
                        false_col = np.zeros([row, int(abs(move[i][0]))])
                        false_col = np.full_like(false_col, False)      
                        if move[i][0] < 0:
                            origin = np.delete(origin, range(int(abs(move[i][0]))), axis=1)
                            origin = np.column_stack([false_col, origin])
                        else:
                            origin = np.delete(origin, range(col - int(abs(move[i][0])), col), axis=1)
                            origin = np.column_stack([origin, false_col])

                    if not np.isnan(move[i][1]) and move[i][1] != 0:
                        false_row = np.zeros([int(abs(move[i][1])), col])
                        false_row = np.full_like(false_row, False)
                        if move[i][1] < 0:
                            origin = np.delete(origin, range(int(abs(move[i][1]))), axis=0)
                            origin = np.row_stack([origin, false_row])
                        else:
                            origin = np.delete(origin, range(row - int(abs(move[i][1])), row), axis=0)
                            origin = np.row_stack([false_row, origin])
                    new_mask[i] = origin

                new_mask = torch.tensor(new_mask)
                pred['instances'].set('pred_boxes', new_box)
                pred['instances'].set('pred_masks', new_mask)
            return should, pred, abs(avg_mov)

        # cal_for_frames
        frame_gen = self._frame_from_video(video)
        # prev = cv2.imread(frames[0])
        # prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        flow_cnt = []
        boxes_save = []
        avg_movement = []
        cnt = 0
        for cnt, frame in enumerate(frame_gen):
            if cnt == 0:
                pre_pred = self.predictor(frame)
                # print("pred:", pred)
                yield process_predictions(frame, self.predictor(frame))
                prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            if cnt == 10:
                flow_visualization(prev, flow)
            
            should, flow_pred, avg_mov = should_send(prev, pre_pred, flow)
            avg_movement.append(avg_mov)
            if should:
                pre_pred = self.predictor(frame)
                boxes_save.append(pre_pred['instances'].get('pred_boxes').tensor)
                yield process_predictions(frame, pre_pred)
            else:
                flow_cnt.append(cnt)
                boxes_save.append(flow_pred['instances'].get('pred_boxes').tensor)
                pre_outframe = process_predictions(frame, flow_pred)
                yield pre_outframe
            prev = curr
        print("\n\n", flow_cnt, "\n\n")
        if cnt != 0:
            print(len(flow_cnt)/cnt)
        print(sorted(avg_movement))
        boxes_save = np.array(boxes_save)
        move_save = np.nan_to_num(np.array(avg_movement))*10
        addr = output_name[6:-4] + "boxes" + str(threshold) +".npz"
        np.savez(addr, boxes = boxes_save)
        addr = output_name[6:-4] + "deviation" + str(threshold) +".npz"
        np.savez(addr, move = move_save)

class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
