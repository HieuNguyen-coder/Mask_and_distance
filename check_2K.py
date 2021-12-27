import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tkinter import * 
from tkinter.ttk import *
from tkinter import messagebox
from tkinter import filedialog
from scipy.spatial import distance
####################################################################################################################################################
mp_face_detection = mp.solutions.face_detection
model = tf.keras.models.load_model('Model\\model_Xception.h5')		
weightsPath = 'D:\\Python\\2K_detection\\Model\\yolov3.weights'
configPath = 'D:\\Python\\2K_detection\\Model\\yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
MIN_DISTANCE = 200
RED = (255, 0, 0)
GREEN = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
####################################################################################################################################################
def makeCenter(root):
	root.update_idletasks()
	width = root.winfo_width()
	height = root.winfo_height()
	x = (root.winfo_screenwidth()//2) - (width//2)
	y = (root.winfo_screenheight()//2) - (height//2)
	root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
####################################################################################################################################################
def get_input_file():
	global input_file
	input_file = filedialog.askopenfilename(initialdir="D:", title = "Select a file", filetypes = (("all files", "*.*"), ("video files", "*.mp4"), ("video files", "*.mov"), ("video files", "*.avi")))
	if len(input_file)!=0: 
		input_label.config(text=input_file, foreground='green')
	else:
		input_label.config(text="Please choose input file", foreground='red')

def get_output_file():
	global output_file
	output_file = filedialog.asksaveasfilename(initialdir='D:', title = "Save file", filetypes = (("all files", "*.*"), ("video files", "*.mp4"), ("video files", "*.mov"), ("video files", "*.avi")))
	if len(output_file)!=0:
		if not (output_file.endswith('.mp4') or output_file.endswith('.mov') or output_file.endswith('.avi')):
			output_file += '.mp4'
		output_label.config(text=output_file, foreground='green')
	else:
		output_label.config(text='Please name output file', foreground='red')
####################################################################################################################################################
def video():
	if len(input_file)>0 and len(output_file)>0:
		check_2K(use_camera=False)
	else:
		messagebox.showerror('Error', 'Please enter both input file and output folder')

def camera():
	check_2K()
####################################################################################################################################################
def find_people(outputs, image, confThreshold=0.5, nmsThreshold=0.5):
	height, width = image.shape[:2]
	people, confs, results = [], [], []
	for output in outputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > confThreshold and classID==0:
				w, h = int(detection[2]*width), int(detection[3]*height)
				x, y = int(detection[0]*width - w/2), int(detection[1]*height - h/2)
				people.append([x, y, w, h])
				confs.append(float(confidence))
	
	indices = cv2.dnn.NMSBoxes(people, confs, confThreshold, nmsThreshold)
	for i in indices:
		results.append(people[i])
		# Try this if the above line causes error
		# results.append(people[i[0]])
	return results

def measureZ(person):
	w, h = person[2], person[3]
	return ((2*3.14*180)/(w+h*360)*1000+3)
####################################################################################################################################################
def check_2K(use_camera=True):
	global input_file, output_file
	if use_camera:
		try:
			cap = cv2.VideoCapture(0)
		except:
			messagebox.showerror('Error', 'Cannot use camera right now\nPlease try again !!!')
	else:
		try:
			cap = cv2.VideoCapture(input_file)
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			output_video = cv2.VideoWriter(output_file, fourcc, 24, 
											(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
		except:
			messagebox.showerror('Error', 'Cannot create video\nPlease try again !!!')

	while cap.isOpened():
		success, image = cap.read()

		# Ignore camera errors
		if not success and use_camera:
			continue

		# Break when the video ends
		if not success and not use_camera:
			break

		with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4) as face_detection:
			H, W = image.shape[0], image.shape[1]
			result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			face_results = face_detection.process(image)

			# If there are faces in the image -> draw bbox and mark
			if face_results.detections:
				for detection in face_results.detections:
					try:
						x = int(detection.location_data.relative_bounding_box.xmin*W)
						y = int(detection.location_data.relative_bounding_box.ymin*H)
						w = int(detection.location_data.relative_bounding_box.width*W)
						h = int(detection.location_data.relative_bounding_box.height*H)
						tmp_image = image[y:y+h, x:x+w]
						tmp_image = cv2.resize(tmp_image, (128, 128))
						tmp_image = np.reshape(tmp_image, (1, 128, 128, 3))/255.0
						# Prediction
						if np.argmax(model.predict(tmp_image)[0]) == 0:
							cv2.putText(result_image, 'No Mask', (x, y-5), FONT, 1, RED, 2)
							cv2.rectangle(result_image, (x, y), (x+w, y+h), RED, 2)
						else:
							cv2.putText(result_image, 'Mask', (x, y-5), FONT, 1, GREEN, 2)
							cv2.rectangle(result_image, (x, y), (x+w, y+h), GREEN, 2)
					except:
						continue

	#=============================================================================================================
		# Distancing
		blob = cv2.dnn.blobFromImage(image, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
		net.setInput(blob)
		layerNames = net.getLayerNames()
		output_names = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
		# Try this if the above line causes error
		#output_names = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
		outputs = net.forward(output_names)

		people = find_people(outputs, image)
		positions = []
		for i in range(len(people)):
			x, y, w, h = people[i]
			positions.append([(x+w)//2, (y+h)//2, measureZ(people[i]),True])
		
		for i in range(len(people)-1):
			for j in range(i+1, len(people)):
				d = distance.euclidean(positions[i][:3], positions[j][:3])
				if d<MIN_DISTANCE:
					positions[i][-1] = False
					positions[j][-1] = False 
		
		for i in range(len(people)):
			x, y, w, h = people[i]
			if positions[i][-1]:
				cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
				cv2.putText(result_image, 'Safe', (x, y-5), FONT, 1, GREEN, 2)
			else:
				cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
				cv2.putText(result_image, 'Not Safe', (x, y-5), FONT, 1, RED, 2)
	#=============================================================================================================
		if use_camera:
			cv2.putText(result_image, "Press 'Esc' to quit", (5, 15), FONT, 1, GREEN, 2)
			cv2.imshow('2K check', cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
			if cv2.waitKey(5) & 0xFF == 27:
				break
		else:
			output_video.write(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
			
	if not use_camera:
		messagebox.showinfo('Info', 'Create video successfully')
####################################################################################################################################################
# Define the window
root = Tk()
root.title('Mask and distance check')
root.geometry("600x250")
makeCenter(root)
root.resizable(width=False, height=False)
label = Label(root, text='2K CHECK', font=('digital-7', 20)).pack(pady=5)

# Input frame to get the input file
input_file = ''
input_frame = Frame(root)
input_button = Button(input_frame, width=20, text = 'Choose input file', command=get_input_file)
input_button.pack(side=LEFT)
input_label = Label(input_frame, width=60)
input_label.pack()
input_frame.pack(pady=10)

# Output frame to get the output folder
output_file = ''
output_frame = Frame(root)
output_button = Button(output_frame, width=20, text = 'Save as', command=get_output_file)
output_button.pack(side=LEFT)
output_label = Label(output_frame, width=60)
output_label.pack()
output_frame.pack(pady=5)

# Create video
create_button = Button(root, text='Create video', width=15, command=video)
create_button.pack(pady=10)
# Use camera
camera_button = Button(root, text='Use camera', width=15, command=camera)
camera_button.pack(pady=20, padx=50, side=LEFT)

root.mainloop()