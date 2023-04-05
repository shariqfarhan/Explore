# The Assignment



1.  OpenCV Yolo:  [SOURCE](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)
      1. Run this above code on your laptop or Colab. 
      2. Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
      3. Run this image through the code above. 
      4. Upload the link to GitHub implementation of this
      5. Upload the annotated image by YOLO. 
2.  Training Custom Dataset on Colab for YoloV3
    1.  Refer to this Colab File:  [LINK](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS)
    2.  Refer to this GitHub  [Repo](https://github.com/theschoolofai/YoloV3)
    3.  Download [this dataset](https://drive.google.com/file/d/1sVSAJgmOhZk6UG7EzmlRjXfkzPxmpmLy/view?usp=sharing) This was annotated by EVA5 Students. Collect and add 25 images for the following 4 classes into the dataset shared:
        *  class names are in custom.names file. 
        *  you must follow exact rules to make sure that you can train the model. Steps are explained in the README.md file on github repo link above.
        *  Once you add your additional 100 images, train the model
    4. Once done:
      *  Download a very small (~10-30sec) video from youtube which shows your classes. 
      *  Use ffmpeg to extract frames from the video. 
      *  Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
      *  Infer on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub. 
      *  python detect.py --conf-three 0.3 --output output_folder_name
      *  Use  ffmpeg  to convert the files in your output folder to video
      *  Upload the video to YouTube. 
      *  Also run the model on 16 images that you have collected (4 for each class)
    5.  Share the link to your GitHub project with the steps mentioned above - 1000 pts (only if all the steps were done, and it resulted in a trained model that you could run on video/images)
    6.  Share the link to your YouTube video - 500 pts
    7.  Share the link of your YouTube video shared on LinkedIn, Instagram, medium, etc! You have no idea how much you'd love people complimenting you! [TOTALLY OPTIONAL] - bonus points
    8.  Share the link to the readme file where we can find the result of your model on YOUR 16 images. - 500 pts

# The Solution

The Assignment involved multiple steps which helps us understand YOLO.

    1. We understand how to detect objects using YOLO - out of the box
    2. We create a custom YOLO detection model, this involved:
          1. Identifying the custom classes which are to be detected
          2. Getting 25 images for each of these classes
          3. We use an annotation tool to help the model learn where the object is in the image
          4. We align the images to the text label and annotation tool output
          5. This is in turn fed as an input to the YOLO base for training
    3. Once trained, we use two levels of testing:
          1. On the Train Dataset
          2. On an unrelated video containing these classes 

Annotation Tool : [MakeSense.ai](https://www.makesense.ai/)

1. The code to identify objects in YOLO (out of the box) can be found [here](https://github.com/shariqfarhan/Explore/blob/master/Assignment_12/YOLO_Predictions.py)
2. The steps for base setup to detect YOLO on custom dataset can be found [here](https://github.com/theschoolofai/YoloV3)
      1. After the base setup, run [this code](https://github.com/shariqfarhan/Explore/blob/master/Assignment_12/YOLO_Custom_Detection.py)
3. The code to get base data to YOLO format can be found [here](https://github.com/shariqfarhan/Explore/blob/master/Assignment_12/utils/base_data_prep.py)


Video Output : [Object Detection](https://youtu.be/oz7feovtlcg)


## Images Annotated by YOLO

Sample Output

![WhatsApp Image 2023-03-18 at 3 31 21 PM (1)](https://user-images.githubusercontent.com/57046534/229267544-c1c53599-78dd-4b85-b3d5-b17f22dbdf95.jpeg)


## YoloV3 Custom Training

For the 2nd part of the assignment I wanted to try checking the model performance on a variety of things but with a common theme. In the above repo, we saw the performance of the model on identifying cartoon characters, through this assignment I wanted to see how the model performs on various landmarks. So I chose 4 different types of landmarks

1. Historical Site ([Charminar](https://en.wikipedia.org/wiki/Charminar))
2. Statue ([The Buddha Statue](https://en.wikipedia.org/wiki/Buddha_Statue_of_Hyderabad))
3. Food ([Hyderabadi Biryani](https://en.wikipedia.org/wiki/Hyderabadi_biryani))
4. Building ([Hitech City](https://en.wikipedia.org/wiki/HITEC_City#:~:text=The%20Hyderabad%20Information%20Technology%20and,in%20Hyderabad%2C%20Telangana%2C%20India.))

The first 3 items in the list had good training images and the model performance was decent in the final video.
But the 4th had limited quality images with lot of noise in the search results. In hindsight, taking a different item would have made this perfect but due to incomplete training data this class couldn't be predicted.

The following images show the output from the custom YOLO detection model.

In addition to the above task, we took a video, split it into multiple frames, ran the custom YOLO detection model on these frames and stitch it back together into 1 output video.

The video used for training can be found [here](https://github.com/shariqfarhan/Explore/tree/master/Assignment_12/static)
The output video can be found on youtube [here](https://youtu.be/oz7feovtlcg)


### YOLO Custom detection output


#### **Charminar**

![image](https://user-images.githubusercontent.com/57046534/229268195-1f9dbc4b-1f60-45ae-9dbe-37ea166c3f6a.png)
![image](https://user-images.githubusercontent.com/57046534/229268209-000b5366-ce2e-44b7-b708-c926360537cb.png)
![image](https://user-images.githubusercontent.com/57046534/229268238-2ea9a563-d737-4d6d-86ed-5143cfa28832.png)


#### Buddha Statue

![image](https://user-images.githubusercontent.com/57046534/229268096-51210c08-40b3-4919-ae34-149cebf06ccb.png)
![image](https://user-images.githubusercontent.com/57046534/229268242-36e2f34e-9b5e-42f6-956f-2758fbf2a098.png)



#### Biryani

![image](https://user-images.githubusercontent.com/57046534/229268165-e4454521-cb32-42b3-8da2-9744eb7e61b1.png)
![image](https://user-images.githubusercontent.com/57046534/229268112-c1bc6e11-86ec-4dc8-a333-a7b556dc59ff.png)




## Train Performance
We trained each of the classes with ~25 images for 50 epochs. Below are the sample images used for training and testing.

![image](https://user-images.githubusercontent.com/57046534/229205395-56276c0a-16d2-42d0-b44e-8ab9e95c21e1.png)

## Train Batch Input
![image](https://user-images.githubusercontent.com/57046534/229205667-0b8d0c61-c2fd-4469-92db-6d71492970fe.png)


## Test Batch Output

![image](https://user-images.githubusercontent.com/57046534/229205573-be63054e-e3ce-46cd-9e48-05345ec44fc0.png)
