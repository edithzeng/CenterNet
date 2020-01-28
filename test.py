import sys
import numpy as np
import cv2
CENTERNET_PATH = './src/lib'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = '/root/temp/CenterNet/models/ctdet_coco_hg.pth'
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --arch {}'.format(TASK, MODEL_PATH, 'hourglass').split(' '))
detector = detector_factory[opt.task](opt)

img = '/root/temp/CenterNet/images/frame0410.jpg' 
ret = detector.run(img)['results']

# save and plot bounding boxes 
im = cv2.imread(img)

for cat, records in ret.items():
    for det in records:

        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        score = det[-1]

        print(f'Class: {cat}')
        print('Bounding box:', det[:-1])
        print('Score:', score)
	
        if int(cat) == 1:
            cv2.rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)

cv2.imwrite('test_frame.jpg', im)
