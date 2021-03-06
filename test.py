import sys
import numpy as np
import cv2
CENTERNET_PATH = './src/lib'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = '/root/temp/CenterNet/models/multi_pose_hg_3x.pth'
# TASK = 'ctdet' # or 'multi_pose' for human pose estimation
TASK = 'multi_pose'
opt = opts().init('{} --load_model {} --arch {}'.format(TASK, MODEL_PATH, 'hourglass').split(' '))
detector = detector_factory[opt.task](opt)

img = '/root/temp/CenterNet/images/frame0410.jpg' 
# video = '/root/temp/CenterNet/data/test.mp4'
ret = detector.run(img)['results']

# save and plot bounding boxes 
im = cv2.imread(img)

if TASK == 'ctdet':
    for cat, records in ret.items():
        for det in records:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            score = det[-1]
            print(f'Class: {cat}')
            print('Bounding box:', det[:-1])
            print('Score:', score)
            if int(cat) == 1:
                cv2.rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
else:
    print(ret)
    sys.exit(0)
    for cat, records in ret.items():
        if int(cat) == 1:
            for det in records:
                for i in range(len(det)-1):
                    center = (int(det[i]), int(det[i+1]))
                    cv2.circle(im, center, 1, (0, 255, 255), 1)
                

cv2.imwrite('test_frame.jpg', im)
