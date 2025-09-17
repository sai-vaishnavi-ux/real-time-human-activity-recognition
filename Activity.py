
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
model = load_model('C:/Users/SAHITHI/Desktop/Project/models/human_activity_model.h5') 
activity_labels = ['Walking', 'Running', 'Sitting', 'Standing', 'Laying', 'Ascending', 'Descending']
video_path = "C:/Users/SAHITHI/Downloads/20444073-uhd_2160_3840_60fps.mp4" 
cap = cv2.VideoCapture(video_path)

scaler = StandardScaler()

prev_gray = None

cnn_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def preprocess_frame(frame):
   
    frame_resized = cv2.resize(frame, (224, 224))

    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    frame_preprocessed = image.img_to_array(frame_rgb)
    frame_preprocessed = np.expand_dims(frame_preprocessed, axis=0)
    frame_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(frame_preprocessed)
    return frame_preprocessed

def extract_features(frame):
    global prev_gray

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean = np.mean(gray_frame)
    skewness = skew(gray_frame.flatten())
    kurt = kurtosis(gray_frame.flatten())
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mean = np.mean(mag) 
        flow_std = np.std(mag)    
    else:
        flow_mean, flow_std = 0, 0 

    prev_gray = gray_frame

    cnn_features = cnn_model.predict(preprocess_frame(frame))
    cnn_features = cnn_features.flatten() 

    features = np.concatenate([cnn_features, [mean, skewness, kurt, flow_mean, flow_std]])
    return features

while True:

    ret, frame = cap.read()

    if not ret:
        print("End of video or failed to grab frame.")
        break

    print(f"Original Frame Shape: {frame.shape}")

    features = extract_features(frame)

    features = features[:561]  
    features = np.pad(features, (0, max(0, 561 - features.size)), 'constant')  
    features = features.reshape(1, 561, 1) 

    features_2d = features.reshape(1, -1) 

    features_2d = scaler.fit_transform(features_2d)

    features = features_2d.reshape(1, 561, 1)  

    predictions = model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)

    predicted_activity = activity_labels[predicted_class[0]]

    cv2.putText(frame, f"Predicted Activity: {predicted_activity}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    cv2.imshow("Activity Recognition", frame)
    cv2.resizeWindow("Activity Recognition", 1000, 1000)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






