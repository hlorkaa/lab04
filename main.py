
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import time

import warnings
warnings.filterwarnings("ignore")

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):

    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
            % (
               capture_width,
               capture_height,
               framerate,
               flip_method,
               display_width,
               display_height,
              )
    )

print('Press 4 to Quit the Application\n')

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

start_loading_time = time.time()
# model = torchvision.models.efficientnet_b0().to(device)
# model.load_state_dict(torch.load('model_state_dict.pth'))
model = torch.load('model.pth')
print("Model loading time: {}".format(time.time()-start_loading_time))

#Open Default Camera
cap = cv2.VideoCapture(0)

img_transforms = transforms.Compose([
    transforms.ToTensor()
])

while(cap.isOpened()):
    start_frame_time = time.time()
    #Take each Frame
    ret, frame = cap.read()

    #Flip Video vertically (180 Degrees)
    #frame = cv2.flip(frame, -180)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    tensor_image = img_transforms(frame)
    batch = torch.unsqueeze(tensor_image, 0).to(device)

    start_recognition_time = time.time()
    output = model(batch)
    print("Recognition time: {}".format(time.time() - start_recognition_time))

    sorted, indices = torch.sort(output, descending=True)

    conclusion = "Not Tomato!"

    if indices[0][0] == 114:
        conclusion = "Tomato!"
    elif indices[0][0] == 128:
        conclusion = "Non-Ripened Tomato!"

    cv2.putText(frame, str(conclusion), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Show video
    cv2.imshow('Cam', frame)


    # Exit if "4" is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 52:  # ord 4
        # Quit
        print('Good Bye!')
        break

    print("Frame processing time: {}".format(time.time() - start_frame_time))

#Release the Cap and Video
cap.release()
cv2.destroyAllWindows()
