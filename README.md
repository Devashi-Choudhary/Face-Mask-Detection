# Face-Mask-Detector
The novel COVID-19 virus has forced us all to rethink how we live our everyday lives while keeping ourselves and others safe. Face masks have emerged as a simple and effective strategy for reducing the virus’s threat and also, there is no efficient face mask detection applications which are now in high demand for transportation means, densely populated areas, residential districts, large-scale manufacturers and other enterprises to ensure safety. Therefore, the goal of the project is to develop face mask detector using deep learning.

# Dependencies
Deep Learning based Face-Mask-Detector architecture uses [OpenCV](https://opencv.org/) (opencv==4.2.0) and Python (python==3.7). The model Convolution Neural Network(CNN) uses [Keras](https://keras.io/) (keras==2.3.1) on [Tensorflow](https://www.tensorflow.org/) (tensorflow>=1.15.2). For face detection in images it uses [caffe based model](https://caffe.berkeleyvision.org/). Also, imutils==0.5.3, numpy==1.18.2, matplotlib==3.2.1, argparse==1.1 are also used.

# How to execute code:

1. You will first have to download the repository and then extract the contents into a folder.
2. Make sure you have the correct version of Python installed on your machine. This code runs on Python 3.6 above.
3. Now, run the following command in your Terminal/Command Prompt to install the libraries required
> `pip install requirements.txt`
4. Now, you can download the dataset from [here](https://drive.google.com/drive/folders/1UGQP83v6gdZXefLAkef1PEjfyjUx0cpY?usp=sharing) and put it in the current folder. The images used in the dataset are real images of people wearing mask i.e. tha dataset doesn't contains morphed masked images. The model is accurately trained and, also the system can therefore be used in real-time applications which require face-mask detection.
5. You also need to download [caffe based face detector model](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/  ) and put it inside face_detection folder.
6. **Training of CNN Model :** Open terminal. Go into the project directory folder and type the following command:
> `python train.py --dataset dataset`
7. **Testing of CNN Model :**  You can download the pretrained model from [here](https://drive.google.com/file/d/1XW62FB60uLaDwFeqOF6qYhaaA8EosoDh/view?usp=sharing) for inference.

For detecting face mask in images, run the following command :
> `python Mask_Detection_in_Image.py --image data/image1.jpg`

For detecting face mask in real-time video stream, run the following command :
> `python Mask_Detection_in_Video.py`

# Results 

1. Accuracy/Loss training curve plot.

![output1](https://github.com/Devashi-Choudhary/Face_Mask_Detection/blob/master/Results/train_loss.png)

2. Mask Detection in Images.

![output2](https://github.com/Devashi-Choudhary/Face_Mask_Detection/blob/master/Results/image1_output.JPG)

![output3](https://github.com/Devashi-Choudhary/Face_Mask_Detection/blob/master/Results/image3_output.JPG)

3. Mask Detection in Real-Time Video Stream.

![Real-Time Video](https://github.com/Devashi-Choudhary/Face_Mask_Detection/blob/master/Results/video_output.mp4)

**Note :** For more details about the implementation, go through [Face-Mask-Detection using Deep Learning](https://medium.com/@Devashi_Choudhary/face-mask-detection-using-deep-learning-83f927654f1e)
