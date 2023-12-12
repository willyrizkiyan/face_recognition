import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import json

# willy_image = face_recognition.load_image_file('photos/Willy/Willy.jpg')
# willy_encoding = face_recognition.face_encodings(willy_image)[0]

# hanan_image = face_recognition.load_image_file('photos/Hanan/Hanan.jpg')
# hanan_encoding = face_recognition.face_encodings(hanan_image)[0]

# athirah_image = face_recognition.load_image_file('photos/Athirah/Athirah.jpg')
# athirah_encoding = face_recognition.face_encodings(athirah_image)[0]

# known_face_encodings = [willy_encoding, hanan_encoding, athirah_encoding]
# known_face_names = ['Willy', 'Hanan', 'Athirah']

# known_face_encodings = [encoding.tolist() for encoding in known_face_encodings]

# encodings_data = {
#     "encodings": known_face_encodings,
#     "names": known_face_names
# }

# with open("face_encodings.json", "r") as file:
#     encodings_data = json.load(file)

# known_face_encodings = encodings_data["encodings"]
# known_face_names = encodings_data["names"]

with open("face_encodings.json", "r") as file:
    all_face_encodings = json.load(file)

known_face_names = []
known_face_encodings = []

for key in all_face_encodings.keys():
    for encoding in all_face_encodings[key]:
        known_face_names.append(key)
        known_face_encodings.append(encoding)

# loaded_face_encodings = {}
# for person, encodings in all_face_encodings.items():
#     loaded_face_encodings[person] = encodings

# known_face_names = list(loaded_face_encodings.keys())
# known_face_encodings = list(loaded_face_encodings.values())

# known_face_encodings = [person["encoding"] for person in people_data]
# known_face_names = [person["name"] for person in people_data]

students = known_face_names.copy()

video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime('%Y-%m-%d')

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

recognized = {}

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ''
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)
            presented = []
            if name not in recognized:
                recognized[name] = True
                if name in students:
                    students.remove(name)
                    presented.append(name)
                    print(presented)
                    # print(students)
                    current_time = now.strftime('%H:%M:%S')
                    lnwriter.writerow([name, current_time])
            # print(name)

            # face_names.append(name)
            # if name in known_face_names:
            #     if name in students:
            #         students.remove(name)
            #         print(students)
            #         current_time = now.strftime('%H-%M-%S')
            #         lnwriter.writerow([name, current_time])

            
            
            top, right, bottom, left = face_locations[0]
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close(f)