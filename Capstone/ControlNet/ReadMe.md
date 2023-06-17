# Controlnet on Pokemon Images

In this Project we implement [ControlNet](https://huggingface.co/lllyasviel/ControlNet-v1-1) on [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5). 

The condition block takes edges and converts them into real images based on a prompt. We train the controlnet model from scratch.

Below are the steps taken to achieve this

1. Data Preparation
2. Process this data and feed into a Pre-Trained Stable Diffusion model
4. Generate Inferences

We trained the model on [pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) dataset and below is a sample output in various stages of training.

## Sample Predictions

| Base Image | Prompt | 5,000 Steps (~24 Epochs) | 10,000 Steps (~48 Epochs)  | 15,000 Steps (~72 Epochs)  | 20,900 Steps (~100 Epochs)|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ![source](https://github.com/shariqfarhan/Explore/assets/57046534/4536b247-f25c-430c-8e2c-d9901370f339) | "a drawing of a green pokemon with red eyes" | ![image](https://github.com/shariqfarhan/Explore/assets/57046534/9cfdd4b0-690b-4098-9307-d1117c5153ba)  | ![image](https://github.com/shariqfarhan/Explore/assets/57046534/b7db9548-d307-443c-80e8-77e1c0b7ff17)  | ![image](https://github.com/shariqfarhan/Explore/assets/57046534/c62e55f4-93e5-43c6-bca9-a9d675018f1a)  | ![image](https://github.com/shariqfarhan/Explore/assets/57046534/87b9c034-b983-466e-b356-5eae42dc59d5)  
| ![image](https://github.com/shariqfarhan/Explore/assets/57046534/d475c5b3-a4da-4320-a44d-d8d9f93e50a2) | "a red and a white ball with angry look on it's face" |  ![image](https://github.com/shariqfarhan/Explore/assets/57046534/ece4fad9-1575-4227-a8fd-7ab29f786926) | ![image](https://github.com/shariqfarhan/Explore/assets/57046534/d1641fc0-0ca9-4707-ba6c-6dd39bc17503) | ![image](https://github.com/shariqfarhan/Explore/assets/57046534/b43fac3d-4ff2-4127-a635-ecaa71bab677) | ![image](https://github.com/shariqfarhan/Explore/assets/57046534/7f97986f-7035-4dc9-9f21-6842648a6c4c) |

### Data Preparation

The processed data used for this project can be found on Hugging Face called [pokemon_canny](https://huggingface.co/datasets/ShariqFarhan/pokemon_canny/tree/main). To create a conditioning image, we use the [canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) function from opencv library. The lambdalabs/pokemon-blip-captions contains images with captions, to this we use [canny](https://github.com/shariqfarhan/Explore/tree/master/Capstone/ControlNet/utilshttps://github.com/shariqfarhan/Explore/tree/master/Capstone/ControlNet/utils) to convert base images to canny edge, which would further be used as conditioning images for training.

The processed files & dataset can be accessed [here](https://huggingface.co/datasets/ShariqFarhan/pokemon_canny/tree/main). 






