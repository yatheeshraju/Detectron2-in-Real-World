from Detector import *

detector = Detector()

# detector.onImage("./testfiles/IMG_0016.jpg")
#detector.onVideo("./testfiles/IMG_0016.mov")
detector.onCamera(0)  # 0 for webcam mostly
##detector.playAudio('testfiles/horn.wav')
#detector.testCamera(6)