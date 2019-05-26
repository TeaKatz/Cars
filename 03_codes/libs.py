import scipy.io
import cv2
import numpy as np


def load_annotations(file_dir):
    """
    input:
            file_dir: is a directory of annotations file
            annotations index:
                0: is box_x1
                1: is box_y1
                2: is box_x2
                3: is box_y2
                4: is class
                5: is filename
    return: tuple of numpy arrays (filenames, boxes, classes)
    """
    mat = scipy.io.loadmat(file_dir)
    filenames = []
    boxes = []
    classes = []
    for anno in mat["annotations"].ravel():
        _filename = anno[5].item()
        _boxe = [anno[0].item(), anno[1].item(), anno[2].item(), anno[3].item()]
        _class = anno[4].item()

        filenames.append(_filename)
        boxes.append(_boxe)
        classes.append(_class)

    return (np.array(filenames), np.array(boxes), np.array(classes))


def load_classes(file_dir):
    """
    input:
            file_dir: is a directory of class file
    return: numpy array of classes
    """
    mat = scipy.io.loadmat(file_dir)
    class_names = []
    for class_name in mat["class_names"].ravel():
        class_names.append(class_name.item())

    return np.array(class_names)


def crop_image(image, box, save_dir=None):
    """
    input:
            image: numpy array of image
            box: numpy array of [x1, y1, x2, y2]
            save_dir: save directory
    return: Cropped image
    """
    # Crop image
    x1, y1, x2, y2 = box
    cropped_image = image[y1:y2, x1:x2]

    # Save image
    if save_dir is not None:
        try:
            cv2.imwrite(save_dir, cropped_image)
        except Exception as e:
            print(e)

    return cropped_image