import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Define 3 draggable boxes (x, y)
boxes = [
    {"pos": [100, 100], "size": 100, "color": (255, 0, 255)},
    {"pos": [300, 150], "size": 100, "color": (0, 255, 0)},
    {"pos": [200, 300], "size": 100, "color": (0, 255, 255)},
]
dragging_box_index = None

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    index_tip = None
    middle_tip = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            index_tip = lm_list[8]    # Index finger tip
            middle_tip = lm_list[12]  # Middle finger tip

            # Check distance between index & middle fingers
            if index_tip and middle_tip:
                distance = ((index_tip[0] - middle_tip[0]) ** 2 + (index_tip[1] - middle_tip[1]) ** 2) ** 0.5

                if distance < 40:
                    for i, box in enumerate(boxes):
                        x, y = box["pos"]
                        size = box["size"]
                        if x < index_tip[0] < x + size and y < index_tip[1] < y + size:
                            dragging_box_index = i
                            break
                else:
                    dragging_box_index = None

    # If dragging, update the position of the dragged box
    if dragging_box_index is not None and index_tip:
        new_x = index_tip[0] - boxes[dragging_box_index]["size"] // 2
        new_y = index_tip[1] - boxes[dragging_box_index]["size"] // 2
        boxes[dragging_box_index]["pos"] = [new_x, new_y]

    # Draw all boxes
    for box in boxes:
        x, y = box["pos"]
        size = box["size"]
        color = box["color"]
        cv2.rectangle(img, (x, y), (x + size, y + size), color, -1)

    cv2.imshow("Virtual Drag and Drop - 3 Boxes", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
