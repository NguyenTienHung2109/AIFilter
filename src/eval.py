from typing import List, Tuple

import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from PIL import Image
import torch
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
import numpy as np
import cv2
import time

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.dlib_datamodule import TransformDataset  # noqa: E402
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. from src import utils)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set root_dir to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)

@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    metric_dict = trainer.callback_metrics

    # for predictions use trainer.predict(...)
    log.info("Starting predictions!")
    # predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    # print("predictions:", len(predictions), type(predictions[0]), predictions[0].shape)
    # for pred, (bx, by) in zip(predictions, datamodule.predict_dataloader()):
    #     print("pred:", pred.shape, "batch:", bx.shape, by.shape)
    #     annotated_image = TransformDataset.annotate_tensor(bx, pred)
    #     print("output_path:", cfg.paths.output_dir + "/eval_result.png")
    #     torchvision.utils.save_image(annotated_image, cfg.paths.output_dir + "/eval_result.png")
    #     break
    # return metric_dict, object_dict

    #1 image
    #test_img = cv2.imread('data/ibug_tiny/helen/trainset/146827737_1.jpg')
    #annotated_image = eval_image(model=model, image_path=test_img)
    #torchvision.utils.save_image(annotated_image, "eval_result.png")
    camera_capture(model=model)
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)

def eval_image(image, model):
    # Make bounding box
    
    imgBB = image
    (h, w) = imgBB.shape[:2]

    face_detector = cv2.dnn.readNetFromCaffe("BBDetection\deploy.prototxt.txt"
                                             , "BBDetection/res10_300x300_ssd_iter_140000.caffemodel")
    blob = cv2.dnn.blobFromImage(imgBB, 1.0)
    face_detector.setInput(blob)
    detections = face_detector.forward()
    startX, startY, endX, endY = 0, 0, 0, 0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # cv2.rectangle(imgBB, (startX, startY), (endX, endY), (0, 0, 255), 2)

    imgBB = imgBB[startY-5:endY+5, startX-5:endX+5] # crop image
    # cv2.imshow("Output", imgBB) #To run in Google Colab, comment out this line Colab notebook
    # #cv2_imshow(image) #To run in Google Colab, uncomment this line
    # cv2.waitKey(0)
    color_converted_img = cv2.cvtColor(imgBB, cv2.COLOR_BGR2RGB)

    transform = Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    input_image = Image.fromarray(color_converted_img)
    input_image = np.asarray(input_image)
    print(input_image.shape)
    input_tensor = transform(image=input_image)
    input_tensor = input_tensor['image'].unsqueeze(0)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    print(input_tensor.shape, output_tensor.shape) #1 3 224 224, 1 68 2
    annotated_image = TransformDataset.annotate_tensor(input_tensor, output_tensor)
    return annotated_image

def camera_capture(model):
    vid = cv2.VideoCapture(0)
    vid.set(3,1280)
    vid.set(4,960)

    cv2.namedWindow("selfie", cv2.WINDOW_NORMAL)
    time1 = 0

    while vid.isOpened():
        ok, frame = vid.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        annotated_image = eval_image(model=model, image_path=frame)
        time2 = time()
        if (time2 - time1) > 0:
            frames_per_second = 1.0/(time2 - time1)
            cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        time1 = time2
        cv2.imshow("selfie", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()