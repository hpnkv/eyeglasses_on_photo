import cv2
import torch
from PIL import Image
from imutils import opencv2matplotlib

from ml_glasses.model import GlassesClassifier
from ml_glasses.transforms import FaceAlignTransform, ToTensor


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 160)
fontScale              = 1
fontColor              = (255, 255, 255)
lineType               = 2


def main():
    model = GlassesClassifier()
    model.load_state_dict(torch.load('checkpoint.pt', map_location=torch.device('cpu')))
    model.train(False)
    align = FaceAlignTransform(detector_model='mmod_human_face_detector.dat',
                               shape_predictor='shape_predictor_68_face_landmarks.dat')
    tensorize = ToTensor()

    winname = 'Do I wear eyeglasses?'
    cv2.namedWindow(winname)
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FPS, 5)

    rval, frame = vc.read()

    while True:
        if frame is not None:
            frame = cv2.resize(frame, (320, 180))

            image = opencv2matplotlib(frame)
            image = Image.fromarray(image)
            image = tensorize(align(image))
            image = image.unsqueeze(0)
            pred = model.forward(image)
            _, labels = torch.max(pred.data, 1)

            if labels.item() == 1:
                cv2.putText(frame, 'Yes!',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
            else:
                cv2.putText(frame, 'No...',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
            cv2.imshow(winname, frame)
        rval, frame = vc.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
