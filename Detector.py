from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
import simpleaudio as sa

import cv2


class Detector:

    def __init__(self):
        self.cfg = get_cfg()
        # load model and weights from model zoo
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
        self.cfg.MODEL.DEVICE = 'cpu'
        self.predictor = DefaultPredictor(self.cfg)

    def playAudio(self, audioPath):
        wave_obj = sa.WaveObject.from_wave_file(audioPath)
        play_obj = wave_obj.play()
        play_obj.wait_done()

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)
        viz = Visualizer(image[:, :, ::-1],
                         metadata=MetadataCatalog.get(
                             self.cfg.DATASETS.TRAIN[0]))
        output = viz.draw_instance_predictions(
            predictions["instances"].to("cpu"))
        cv2.imshow("output", output.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    def onVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            video = frame
            predictions = self.predictor(video)
            viz = Visualizer(video[:, :, ::-1],
                             metadata=MetadataCatalog.get(
                                 self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_instance_predictions(
                predictions["instances"].to("cpu"))
            cv2.imshow("output", output.get_image()[:, :, ::-1])
            cv2.imshow('window-name', video)
            count = count + 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(0)

    def onCamera(self, camNumber):
        cap = cv2.VideoCapture(camNumber)
        cap.set(cv2.CAP_PROP_FPS, 1)
        count = 0
        percent = 100
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            width = int(frame.shape[1] * percent / 100)
            height = int(frame.shape[0] * percent / 100)
            dim = (width, height)
            video = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            predictions = self.predictor(video)
            viz = Visualizer(video[:, :, ::-1],
                             metadata=MetadataCatalog.get(
                                 self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_instance_predictions(
                predictions["instances"].to("cpu"))
            detected_class_indexes = predictions["instances"].pred_classes
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            class_catalog = metadata.thing_classes

            for idx in range(len(detected_class_indexes)):
                class_name = class_catalog[detected_class_indexes[idx]]
                print(class_name)
                if (class_name == "bird"):
                    self.playAudio('testfiles/horn.wav')

            cv2.imshow("output", output.get_image()[:, :, ::-1])
            cv2.imshow('window-name', video)
            count = count + 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(0)
