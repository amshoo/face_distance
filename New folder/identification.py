import face_recognition as fr
from PIL import Image, ImageDraw
from datetime import datetime
import datetime

#load image and encode face
image_of_bill = fr.load_image_file('./img/known/sheshe.jpg')
bill_face_encoding = fr.face_encodings(image_of_bill)[0]

image_of_steve = fr.load_image_file('./img/known/goni.jpg')
steve_face_encoding = fr.face_encodings(image_of_steve)[0]

image_of_elon = fr.load_image_file('./img/known/amshi.jpg')
elon_face_encoding = fr.face_encodings(image_of_elon)[0]


#  Create arrays of encodings and names
known_face_encodings = [
  bill_face_encoding,
  steve_face_encoding,
  elon_face_encoding
]

known_face_names = [
  "sheshe",
  "goni",
  "amshi"
]



# Load test image to find faces in
test_image = fr.load_image_file('./img/groups/gp.jpg')


# Find faces in test image
face_locations = fr.face_locations(test_image)
face_encodings = fr.face_encodings(test_image, face_locations)


# Convert to PIL format and  Create a ImageDraw instance
pil_image = Image.fromarray(test_image)
draw = ImageDraw.Draw(pil_image)




# Loop through faces in test image
attendance=[]
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_face_encodings, face_encoding)
    
    name = "Unknown Person"
    # If match
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    #save name    
    time = datetime.datetime.now().strftime("%Y-%B-%d %A %H:%M")
    name_time = name, time
    if name_time not in attendance:
        attendancee = attendance.append(name_time)


#attendance
import pandas as pd
df = pd.DataFrame.from_records(attendance,columns=['Name','Date_Time'])
df





