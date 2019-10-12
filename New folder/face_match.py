import face_recognition as fr


#load known image and encode face
image_of_bill = fr.load_image_file('./img/known/Bill Gates.jpg')
known_face_encoding = fr.face_encodings(image_of_bill)[0]

#load unknown image and encode face
unknown_image = fr.load_image_file('%s')
unknown_face_encoding = fr.face_encodings(unknown_image)[0]
#./img/unknown/d-trump.jpg

# Compare faces
results = fr.compare_faces([known_face_encoding], unknown_face_encoding)


if results[0]:
    print('This is Bill Gates')
else:
    print('This is NOT Bill Gates')


