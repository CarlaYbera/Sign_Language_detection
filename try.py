from gtts import gTTS
import os
from ultralytics import YOLO
import cv2

def speak_text(text):
    # Generate speech from text using gTTS
    tts = gTTS(text=text, lang='en')
    tts.save("temp.mp3")  # Save speech to a temporary file

    # Play the speech using a media player
    os.system("mpg321 temp.mp3")  # Adjust the command based on your system and preferred media player

    # Clean up the temporary file
    os.remove("temp.mp3")

model = YOLO('best.pt')

current_sentence = []

results = model.predict(source="0", save=False, imgsz=640, conf=0.09, show=True, stream=True)
for r in results:
    orig_img = r.orig_img
    boxes = r.pred[0]['xyxy'] if 'pred' in r.__dict__ and r.pred is not None else []  
    
    detected_words = []
    for box in boxes:
        class_id = int(box[5])
        class_label = model.names[class_id]
        print("Detected Word:", class_label) 
        detected_words.append(class_label)

    for word in detected_words:
        if word.lower() not in [w.lower() for w in current_sentence]:
            current_sentence.append(word)
            print("Added to current_sentence:", word)  

   
    if len(current_sentence) > 5:
        current_sentence = current_sentence[-5:]

    subtitle_text = ' '.join(current_sentence)

    print("Subtitle Text:", subtitle_text)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 1
    text_size = cv2.getTextSize(subtitle_text, font, font_scale, font_thickness)[0]
    text_x = 10
    text_y = orig_img.shape[0] - 20  
    cv2.putText(orig_img, subtitle_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

    cv2.imshow('YOLO', orig_img)

    print("Detected Words:", detected_words)
    print("Current Sentence:", current_sentence)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cv2.destroyAllWindows()
