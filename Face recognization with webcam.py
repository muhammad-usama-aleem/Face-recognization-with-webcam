import os
import numpy as np
import cv2
import face_recognition as fg


os.chdir("C:/Users/abdul/OneDrive/Pictures/New folder")
usama_img = fg.load_image_file("usama.jpg")
junaid_img = fg.load_image_file("junaid.jpg")
mufti_img = fg.load_image_file("mufti.jpg")

usama_encoding = fg.face_encodings(usama_img)[0]
junaid_encoding = fg.face_encodings(junaid_img)[0]
mufti_encoding = fg.face_encodings(mufti_img)[0]
known_encoding = [usama_encoding, junaid_encoding, mufti_encoding]
known_img_name = ["Usama", "Junaid", "Mufti"]
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    if process_this_frame:
        face_locations = fg.face_locations(rgb_small_frame)
        face_encodings = fg.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = fg.compare_faces(known_encoding, face_encoding)
            name = "unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_img_name[first_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame
    for (top , right , bottom , left) , name in zip(face_locations , face_names) :
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame , (left , top) , (right , bottom) , (0 , 0 , 255) , 2)
        cv2.rectangle(frame , (left , bottom - 35) , (right , bottom) , (0 , 0 , 255) , cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame , name , (left + 6 , bottom - 6) , font , 0.5 , (255 , 255 , 255) , 1)
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
