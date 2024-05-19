import tellopy
import av
import cv2
import mediapipe as mp
import time
import numpy as np

def connect_tello():
    drone = tellopy.Tello()
    connected = False
    while not connected:
        try:
            drone.connect()
            drone.wait_for_connection(60.0)
            connected = True
        except Exception as e:
            print(f"Connection failed: {e}. Retrying...")
            time.sleep(5)  # Wait before retrying
    return drone

def find_distance(point1, point2, lmList):
    x1, y1 = lmList[point1][1:]
    x2, y2 = lmList[point2][1:]
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Connect to the Tello drone
drone = connect_tello()

# Open the Tello video stream
container = None
while container is None:
    try:
        container = av.open(drone.get_video_stream())
    except Exception as e:
        print(f"Failed to open video stream: {e}. Retrying...")
        time.sleep(5)  # Wait before retrying

# Initialize MediaPipe and other variables for pose detection
zombie = False
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

pTime = 0

# Process frames from the Tello drone
for frame in container.decode(video=0):
    image = frame.to_ndarray(format="bgr24")
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    lmList = []

    if results.pose_landmarks:
        mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    image = cv2.flip(image, 1)
    cv2.putText(image, str(int(fps)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    # Display the image
    cv2.imshow("Tello Feed", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # DOING CALCS
    try:
        armdist = find_distance(15, 16, lmList)
        if armdist < 200:
            print("🚨 ZOMBIE TIME")
            zombie = True
        else:
            print("No zombies")
    except:
        pass

cv2.destroyAllWindows()
drone.quit()
