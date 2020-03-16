import cv2
import torch
from PIL import Image
from imutils import opencv2matplotlib, face_utils

from ml_glasses.model import GlassesClassifier
from ml_glasses.transforms import FaceAlignTransform, ToTensor


FONT = cv2.FONT_HERSHEY_SIMPLEX
BOTTOM_LEFT_CORNER = (10, 160)
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
LINE_TYPE = 2


def put_text(frame, position, text):
    cv2.putText(frame, text,
                position,
                FONT,
                FONT_SCALE,
                FONT_COLOR,
                LINE_TYPE)


def main():
    model = GlassesClassifier()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load('checkpoint.pt', map_location=device))
    model.train(False)
    align = FaceAlignTransform(detector_model='mmod_human_face_detector.dat',
                               shape_predictor='shape_predictor_5_face_landmarks.dat')
    tensorize = ToTensor()

    winname = 'Am I wearing eyeglasses?'
    cv2.namedWindow(winname)
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FPS, 60)

    rval, frame = vc.read()

    while True:
        if frame is not None:
            frame = cv2.resize(frame, (320, 180))

            image = opencv2matplotlib(frame)
            bboxes = align.bounding_boxes(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            image = Image.fromarray(image)
            image = tensorize(align(image))
            image = image.unsqueeze(0)
            pred = model.forward(image)
            _, labels = torch.max(pred.data, 1)

            if labels.item() == 1:
                put_text(frame, BOTTOM_LEFT_CORNER, 'Yes!')
            else:
                put_text(frame, BOTTOM_LEFT_CORNER, 'No...')
            for bbox in bboxes:
                (x, y, w, h) = face_utils.rect_to_bb(bbox.rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(winname, frame)
        rval, frame = vc.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
