from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo

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