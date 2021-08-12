# ART-GAN

In this project, we attempt to explore building Generative Adversarial Networks (GANs) for the purpose of generating new images of artworks. 

### Problem Statement

We are interested in attempting to answer a few questions through the course of this project.

First and most importantly, we want to understand if GANs can help us generate new data in the field of artworks. This can be extended and explored more deeply in the future whereby artists can use GANs for inspiration in creating new works of art similar to their style or a combination of styles from different artists.

Secondly, we want also want to gain a better understanding of GANs and to better understand how to optimize them for the best results.

### Gentle Introduction to GANs

In 2014, Ian Goodfellow, currently employed at Apple Inc. as its director of machine learning in the Special Projects Group, and a few other researchers released a paper on [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661). The paper proposed a new framework for estimating generative models via an adversarial process, in whcih 2 models are simultaneously trained: a generative model that captures the data distribution and a discriminative model that estimates the probability that a sample came from the training data rather than the generative model.

This framework corresponds to a minimax two-player game. The generative model is attempting to maximize the probability of the discriminative model making a mistake and the discriminative model is attempting to minimize its discriminating mistakes. In the space of arbituary models, a unique solution exists where the probability of the discriminative model classifying the sample as coming from the training data and the generative model is half.

A simple framework to think about the entire architecure is as such:
1. We use the generative model to generate X amount of samples
2. We randomly sample X amount of inputs from our training data
3. We feed a batch size of 2X samples into our discriminator to tell us which are real and which are generated
4. The generative model uses this information to generate better samples and the process repeats
5. The model reaches its 'solution' when the discriminator is unable to tell us whether the sample is real or generated (i.e. a 0.5 probability for both cases)

The entire system can be trained with backpropagation.

### Data Generation, Exploration and Augmentation

For this project, we are working with [color field paintings](https://www.tate.org.uk/art/art-terms/c/colour-field-painting). This is the style of art popularized by artists such as Mark Rothko and Jackson Pollock. Color field painting is a style of abstract painting that emerged in New York City during the 1940s and 1950s. It was inspired by European modernism and closely related to abstract expressionism, while many of its notable early proponents were among the pioneering abstract expressionists. Color field is characterized primarily by large fields of flat, solid color spread across or stained into the canvas creating areas of unbroken surface and a flat picture plane. The movement places less emphasis on gesture, brushstrokes and action in favour of an overall consistency of form and process. In color field painting "color is freed from objective context and becomes the subject in itself."

This was also the reason why this genre of art was selected. It allows for simple assessment of the end product given that the style of art is distinct. The style also is 2 dimensional, simplyfing the process and removing picture depth out of the equation. We will be looking out for tell-tale signs such as solid colors creating unbroken surfaces.

The training set or actual paintings were scraped from Wikiart with the help of Selenium. The code can be found in the notebook titled 'Web Image Scraping (Wikiart). After exploring the website and selected artists, the amount of images scraped was 940. This was due to the fact this genre of art is rather specific and unlike expressionist works in general as an example. 

Here are some examples of paintings from the scraped data:
![image](https://user-images.githubusercontent.com/49399188/128220944-4d8ddb51-dfb4-46f1-bbd1-b5707cd25566.png)

The images were cleaned as a handful were actually just photographs of installations rather than the artworks themselves:
![image](https://user-images.githubusercontent.com/49399188/128221261-fcaf44ad-a4b9-49a0-b09d-e54105f008da.png)

The next step would be to prep and augment the images. We want to grow the dataset so that our models have more data to train on. The steps taken were:
1. Resizing the images to 64 x 64
2. Rotating 90, 180 and 270 degrees
3. Flipping left-right and top-bottom

![image](https://user-images.githubusercontent.com/49399188/128221527-5ae6493a-aa4a-40ea-a55c-75858a4e4db6.png)

We managed to grow the dataset but another round of cleaning was done to remove some images that lost most of their picture through resizing (narrow and tall paintings) and also paintings that were just a solid block of colour mostly as these would provide little information for models to train on. We and now are left with a total of **5,532 images**. The full code can be found in the notebook 'Image Augmentation'.

### Model Structure

A discriminator and generator model was built, these are both CNN models. The generator takes input as a latent space of Gaussian distributed values as its 'noise'. LeakyReLU activation was used in most layers with the exception of the last layer of the generator (tanh) and the last layer of the discriminator (sigmoid). Adam optimiser was used as well. No pooling layers were used but instead, strided convolutions. This is slightly costlier to compute but more parameters can be learned. 

The model was run for 150 epochs and was stable throughout the training process. Training for the discriminator was done independently and the real and fake inputs were seperated. Both discriminator and generator models were then stacked with the help of another CNN model and the weights of the discriminator were frozen and only the generator was trained. We did not want to overtrain the discriminator on 'fake samples'

The training process was roughly as follows:
- Sample half batch of real images  train discriminator
- Generate half batch of fake images  train discriminator
- Generate fake images, label as real  Feed into combined model and train (only generator is trained)
- Save plots and models every 10 epochs
- Train for 150 epochs

Some extensions were made to the model:
- Include batch normalization in generator
- Include batch normalization in both generator and discriminator
- Train model on curated dataset

### Model Evaluation
**Model 1**: The baseline model took approximately 2 days to run through 150 epochs. Training losses did not fluctuate much, the discriminator loss hovered from 0 to 1 and generator loss generally was in the range of 2 to 5. Model accuracy did improve through the first few epochs but stayed at 90+% over most epochs. However the results still gradually improved as shown below:

![image](https://user-images.githubusercontent.com/49399188/128222514-b682bd03-80c5-43e0-b6fd-e1a9885537af.png)

**Model 2**: Model 2 with batch normalisation layers in the generator took slightly shorter to run, approximately 1.5 days. Batch normalisation in the generator did help stabalise training and improve computing. Results also seem to be better.

![image](https://user-images.githubusercontent.com/49399188/128222619-177a42e3-c7c4-4f57-8b91-b58af1d323e7.png)

**Model 3**: Model 3 included batch normalisation in both the discriminator and generator and did not perform well at all. Batch normalisation in the discriminator affected training and the model losses hit 0 after ~ 3 epochs. My assumption is that batch normalisation hindered the training of the discriminator. The discriminator would benefit from varied inputs but perhaps batch normalisation oversimplified the training data. This could also be attributed to the nature of the small batch size and small training data. However, there was no similar case that I could reference to and this could be a possible expansion of the project to understand clearly how batch normalisation works.

**Model 4**: Model 4 was trained on a curated dataset of images. It is noted that some of the pictures can be rather messy and noisy. We streamlined the dataset to include only pictures that have solid blocks of colours and clear distinctions. The dataset was shrunk to 1,656 images. As for model 4, the training was much faster due to the nature of the small dataset. Training took approximately 4 hours. We saw that the generated images were more similar and more uniform. However, this hindered the creative potential of the models and they were somewhat restrained by the data fed to them.

![image](https://user-images.githubusercontent.com/49399188/128222861-13f447f7-cd2b-43de-8320-1f8d5dd2c278.png)

### Results

Model 3 was thrown out the window as it was of little use. 4 pictures were generated from models 1,2 and 4 and then passed through a pre-trained super resolution GAN (ESRGAN) to help enhance the images. Samples of how ESRGAN works:
![image](https://user-images.githubusercontent.com/49399188/128223492-ec18a010-784d-4eb8-81e7-ab39897a1ba4.png)

The final results are as such:
![image](https://user-images.githubusercontent.com/49399188/128223054-af94683c-a8f9-4a01-966b-38e561a60659.png)

Drawing some comparisons from the originals to some of the results:
![image](https://user-images.githubusercontent.com/49399188/128223105-d1277518-e73f-4bf1-8274-4cb50209ae70.png)

The results show that GANs can indeed be used in the field of artworks and there is much potentially is using a system to 'aid' our learning of art in different styles and eras. GANs are a powerful tool in generating new data for various fields and this is just one of them. 

Some other takeaways from the project are also some tips and tricks on how to optimise GANs:
* Quality of dataset is most important, including size and variation
* Gaussian-distributed values as latent space for generator input
* LeakyReLU activation layers
* Use strided convolutions instead of pooling layers
* Adam optimiser
* Seperate real and fake images for discriminator training
* Batch normalisation in generator
* Constantly monitor training progress

### Further Extensions

1. Optimising training time even more
2. Train for more epochs
3. Train model on brand new segment of art




