import cv2

def main():
    # Load the models
    face_cascade, age_model_path, gender_model_path = load_models()

    # Load the pre-trained models
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', age_model_path)
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', gender_model_path)

    # Open the video capture
    cap = cv2.VideoCapture(0)

    # Process video frames
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Break the loop if no frame is captured
        if not ret:
            break

        # Detect faces and estimate age and gender
        frame = detect_age_gender(frame, face_cascade, age_net, gender_net)

        # Display the frame
        cv2.imshow('Age and Gender Estimation', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
