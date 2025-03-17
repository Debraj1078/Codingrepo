import cv2
import dlib
import numpy as np

def extract_head(image_path, output_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print(f"No face detected in {image_path}.")
        return

    x, y, w, h = faces[0]
    head = image[y:y+h, x:x+w]
    cv2.imwrite(output_path, head)
    print(f"Head extracted and saved to {output_path}")

def extract_lips(image_path, output_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 0:
        print(f"No face detected in {image_path}.")
        return

    for face in faces:
        landmarks = predictor(gray, face)

        lip_points = []
        for i in range(48, 68):
            lip_points.append((landmarks.part(i).x, landmarks.part(i).y))
        lip_points = np.array(lip_points)

        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [lip_points], 255)

        lip_image = cv2.bitwise_and(image, image, mask=mask)
        (x, y, w, h) = cv2.boundingRect(lip_points)
        cropped_lip = lip_image[y:y+h, x:x+w]

        scale_factor = 4
        lip = cv2.resize(cropped_lip, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(output_path, lip)
        print(f"Lips extracted and saved to {output_path}")

for num in range(1, 26):
    input_image_paths_head = [
        f'C:\\Internship\\{num:03}\\{num:03}_1.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_2.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_3.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_4.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_5.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_6.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_7.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_8.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_9.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_10.jpeg'
    ]

    output_image_paths_head = [
        f'C:\\Internship\\{num:03}_face\\{num:03}_1.jpg',
        f'C:\\Internship\\{num:03}_face\\{num:03}_2.jpg',
        f'C:\\Internship\\{num:03}_face\\{num:03}_3.jpg',
        f'C:\\Internship\\{num:03}_face\\{num:03}_4.jpg',
        f'C:\\Internship\\{num:03}_face\\{num:03}_5.jpg',
        f'C:\\Internship\\{num:03}_face\\{num:03}_6.jpg',
        f'C:\\Internship\\{num:03}_face\\{num:03}_7.jpg',
        f'C:\\Internship\\{num:03}_face\\{num:03}_8.jpg',
        f'C:\\Internship\\{num:03}_face\\{num:03}_9.jpg',
        f'C:\\Internship\\{num:03}_face\\{num:03}_10.jpg'
    ]

    input_image_paths_lip = [
        f'C:\\Internship\\{num:03}\\{num:03}_1.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_2.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_3.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_4.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_5.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_6.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_7.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_8.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_9.jpeg',
        f'C:\\Internship\\{num:03}\\{num:03}_10.jpeg'
    ]

    output_image_paths_lip = [
        f'C:\\Internship\\{num:03}_lips\\{num:03}_1.jpg',
        f'C:\\Internship\\{num:03}_lips\\{num:03}_2.jpg',
        f'C:\\Internship\\{num:03}_lips\\{num:03}_3.jpg',
        f'C:\\Internship\\{num:03}_lips\\{num:03}_4.jpg',
        f'C:\\Internship\\{num:03}_lips\\{num:03}_5.jpg',
        f'C:\\Internship\\{num:03}_lips\\{num:03}_6.jpg',
        f'C:\\Internship\\{num:03}_lips\\{num:03}_7.jpg',
        f'C:\\Internship\\{num:03}_lips\\{num:03}_8.jpg',
        f'C:\\Internship\\{num:03}_lips\\{num:03}_9.jpg',
        f'C:\\Internship\\{num:03}_lips\\{num:03}_10.jpg'
    ]

    for input_path, output_path in zip(input_image_paths_head, output_image_paths_head):
        extract_head(input_path, output_path)

    for input_path, output_path in zip(input_image_paths_lip, output_image_paths_lip):
        extract_lips(input_path, output_path)
