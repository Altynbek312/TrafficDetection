import numpy as np
import argparse
import imutils
import time
import cv2	
import os
import glob
from sort import *
import matplotlib.pyplot as plt

LIMIT = 1500
# Counting lines
#line = [(40, 350), (470, 350)]
#line2 = [(560, 350), (940, 350)]

#Night
#line = [(107, 276), (364, 395)]
#line2 = [(400, 428), (750, 540)]

line1 = [(300, 328), (650, 240)]
line2 = [(400, 428), (750, 540)]
drawing = False
point1 = ()
point2 = ()
point12 = ()
point22 = ()

def mouse_drawing(event, x, y, flags, params):
    global point1, point2, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing is False:
            drawing = True
            point1 = (x, y)
        else:
            drawing = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            point2 = (x, y)

def mouse_drawing2(event, x, y, flags, params):
    global point12, point22, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing is False:
            drawing = True
            point12 = (x, y)
        else:
            drawing = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            point22 = (x, y)

cap = cv2.VideoCapture("./input/Input.mp4")
#"./input/Input.mp4"
cv2.namedWindow("Direction1")
cv2.setMouseCallback("Direction1", mouse_drawing)
_,frame = cap.read()
vidheight, vidwidth = frame.shape[:2]
cv2.putText(frame, "1.Draw a line perpendicular to cars moving in the same direction.", (int(vidheight/15), int(vidwidth/30)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
while True:
    #frame = framo

    if point1 and point2:
        #if key == ord('d'):
        cv2.line(frame, point1, point2, (0, 0, 255))

    cv2.imshow("Direction1", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
#
cap = cv2.VideoCapture("./input/Input.mp4")
cv2.namedWindow("Direction2")
cv2.setMouseCallback("Direction2", mouse_drawing2)
cv2.putText(frame, "2.Draw a line perpendicular to cars moving in different direction.", (int(vidheight/15), int(vidwidth/20)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

while True:
    if point12 and point22:
        #if key == ord('d'):
        cv2.line(frame, point12, point22, (0, 255, 0))

    cv2.imshow("Direction2", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
 
# Clean folder
def Clear() : 
	files = glob.glob('output/*.png')
	for f in files:
		os.remove(f)

# argument object for input parameters
def ParseArgs() :
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input",  default='input\Input.mp4', help="path to input video")
	ap.add_argument("-o", "--output", default='output\TrafficDetected.mp4', help="path to output video")
	ap.add_argument("-y", "--yolo", default='config', help="base path to YOLO directory")
	ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.4, help="threshold when applyong non-maxima suppression")
	args = vars(ap.parse_args())
	return args

# Check intersection with counting line
def Intersect(A,B,C,D):
	return CCW(A,C,D) != CCW(B,C,D) and CCW(A,B,C) != CCW(A,B,D)

def CCW(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Initialize network and get video
def Init(args) : 
	labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# Colors for frames
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

	weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
	print("Loading YOLO...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	vs = cv2.VideoCapture(args["input"])
	
	

	return LABELS , COLORS , net, ln , vs

# Process video
def ParseVideo(args , LABELS , COLORS , net, ln , vs) : 
	tracker = Sort()
	memory = {}

	writer = None
	(W, H) = (None, None)
	
	frameIndex = 0

	# Vehicles counter
	counter = 0
	counter2  = 0
	# Results graph
	sec_x = []
	sec_y_to = []
	sec_y_from = []
	frame_x = []
	frame_y_to = []
	frame_y_from = []
	#Check whether you have cv2 or cv
	try:
		if imutils.is_cv2() == True :
			prop = cv2.cv.CV_CAP_PROP_FRAME_COUNTf
		else :
			prop = cv2.CAP_PROP_FRAME_COUNT
		
		total = int(vs.get(prop))
		print("{} total frames in video".format(total))
	except:
		print("Could not determine # of frames in video")
		total = -1
		return

	succes,roivid = vs.read()
	
	vidheight, vidwidth = roivid.shape[:2]
	cv2.putText(roivid, "Select a region where cars will be detected",(int(vidheight / 15), int(vidwidth / 20)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
	cv2.imwrite("frame.jpg", roivid)

	im2 = cv2.imread("frame.jpg")
	roi = cv2.selectROI(im2)
	cv2.destroyAllWindows()
	while True:
		
		#Get frames
		(grabbed, orig) = vs.read() 
		#frame = orig[500:700, 0:1200]
		
		
		mask = np.zeros(orig.shape[0:2], dtype=np.uint8)
		#points = np.array([[[0,720],[0,500],[400,320],[930,320],[1280,500],[1280,720]]])#normal
		points = np.array([[int(roi[0]),int(roi[1]+roi[3])],[int(roi[0]),int(roi[1])],[int(roi[0]+roi[2]),int(roi[1])],[int(roi[0]+roi[2]),int(roi[1]+roi[3])]])
	 
		#method 1 smooth region
		
		cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
		frame = cv2.bitwise_and(orig,orig,mask = mask)

		if not grabbed:
			break
		
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		#Blobs for input to CNN
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		net.setInput(blob)
		
		
		start = time.time()
		layers = net.forward(ln)
		end = time.time()

		boundaries = []
		confidences = []
		classIDs = []
		classname = []

		for layer in layers :
			for detection in layer :
				
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				if confidence > args["confidence"]:
					# Frames center
					square = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = square.astype("int")

					# Frames upper left corner
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boundaries.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
					#if counter<1:
				   
					
					#classname.append(LABELS[classID])

		idxs = cv2.dnn.NMSBoxes(boundaries, confidences, args["confidence"], args["threshold"])
		objects = []
		
		if len(idxs) > 0 :
			for i in idxs.flatten():
				(x, y) = (boundaries[i][0], boundaries[i][1])
				(w, h) = (boundaries[i][2], boundaries[i][3])
				objects.append([x, y, x+w, y+h, confidences[i]])

		np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
		objects = np.asarray(objects)
		tracks = tracker.update(objects)

		boundaries = []
		IDs = []
		prev = memory.copy()
		memory = {}

		for track in tracks:
			boundaries.append([track[0], track[1], track[2], track[3]])
			IDs.append(int(track[4]))
			memory[IDs[-1]] = boundaries[-1]

		if len(boundaries) > 0:
			i = int(0)
			for square in boundaries:
				# upper left corner
				(x, y) = (int(square[0]), int(square[1]))
				# upper left corner
				(w, h) = (int(square[2]), int(square[3]))

				
				color = [int(c) for c in COLORS[IDs[i] % len(COLORS)]]
				# Framing box
				cv2.rectangle(frame, (x, y), (w, h), color, 2)

				if IDs[i] in prev:
					previous_box = prev[IDs[i]]
					
					(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
					(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
					
					p0 = (int(x + (w - x) / 2) , int(y + (h - y) / 2))
					p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
					
					cv2.line(frame, p0, p1, color, 5)
					
					# Check intersection with counting line
					if Intersect(p0, p1, point1, point2):
						counter += 1
					if Intersect(p0, p1, point12, point22):
						counter2 += 1 

				# Box description
				text = "{},{:.2f},{}".format(IDs[i],confidences[i],LABELS[classIDs[i]])
				cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN,1, color, 1)
				#classname.append(LABELS[classID])
				
				#classIDs[i] = classIDs[i]/2
				#classIDs.astype("int")
				
				#text = "{} {:.2f}".format(IDs[i],confidences[i])
				#cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN,2, color, 2)
				
				i += 1

		# draw line
		frame = cv2.bitwise_or(frame,orig)
		cv2.line(frame,point1,point2, (0, 0, 255), 3)
		cv2.line(frame, point12, point22, (0, 255, 0), 3)
		#line1[0], line1[1],
		# Draw car counters on video
		
		frame = cv2.bitwise_or(frame,orig)
		frame = cv2.polylines(frame,[points],True,(255,255,0), 2)
		cv2.putText(frame, str(counter), (int(vidwidth/10),int(vidheight/5)), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 0, 255), 7)
		cv2.putText(frame, str(counter2), (int(vidwidth/1.5),int(vidheight/5)), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 0), 7)
		
		height, width
		# Save frame as image

		cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

		if writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MPEG")
			writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)
			if total > 0:
				elap = (end - start)
				print("Single frame took {:.4f} seconds".format(elap))
				print("Total time to finish: {:.4f}".format(elap * total))

		writer.write(frame)

		frameIndex += 1
		
		frame_x.append(frameIndex)
		frame_y_to.append(counter)
		frame_y_from.append(counter2)

		if frameIndex % 30 == 0 : 
			sec_x.append(frameIndex // 30)
			sec_y_to.append(counter)
			sec_y_from.append(counter2)

		
		if frameIndex >= LIMIT:
			break

	print(" Cleaning up...")
	
	writer.release()
	vs.release()
	
	return sec_x , sec_y_to , sec_y_from , frame_x , frame_y_to , frame_y_from

def DrawGraf(sec_x , sec_y_to , sec_y_from , frame_x , frame_y_to , frame_y_from):
	
	print('Drawing graf...')
	
	plt.subplot(211)
	plt.plot(sec_x , sec_y_to , 'y--' , sec_x , sec_y_from , 'm--' )
	plt.ylabel('Number of vehicles')
	plt.xlabel('Number of Seconds')
	
	plt.subplot(212)
	plt.plot(frame_x , frame_y_to , 'y--' , frame_x , frame_y_from , 'm--' )
	plt.ylabel('Number of vehicles')
	plt.xlabel('Number of Frames')
	
	plt.show()

def main() :

	Clear()
	args = ParseArgs()
	LABELS , COLORS , net, ln , vs = Init(args)
	sec_x , sec_y_to , sec_y_from , frame_x , frame_y_to , frame_y_from = ParseVideo(args , LABELS , COLORS , net, ln , vs)
	DrawGraf(sec_x , sec_y_to , sec_y_from , frame_x , frame_y_to , frame_y_from)	 
if __name__ == '__main__':
	main()