# A.I. Terminology 

**Artificial Intelligence:** Broad term with cultural significance, ever-shifting and referring to applied, theoretical, and fantastical concepts of Computational Intelligence.

**Machine Learning** is a broad field that explores the design and application of computational systems that are capable of rational problem-solving and/or behavior without explicit instruction (meaning they can adapt to and infer from data) 

All applied ML development focuses on algorithmic and statistical modeling of limited problem-spaces (Artificial Narrow Intelligence). 

## Models, methods, and training, oh my!
Okay, so that is what Machine Learning is about, but what is ***it***?

The final product of a Machine Learning development process is often a **Model**. A model is not a stand alone application or tool, rather it is some code (the core of which is often a single file) that takes specific inputs and parameters (image data, audio data, numerical data, text, etc.) and spits out a specific output (labeled imagery, filtered audio, numerical data groupings, translated text, etc). 

The only difference between a ML model and other forms of code, is that instead of being explicitly written by a human, ML models are ***trained*** (this approach also results in code that looks and behaves very differently under the hood).

Though ML models are often refered to as AI, they are still intentionally crafted by their designers- the model structure, algorithms, parameters, and training data are all curated and often invented by the model's author or other humans. The dataset used in training a model is just as important as the model architecture itself, so you may see the same model architecture trained on a different dataset referred to with a unique name or used for a completely different application.

Okay, so **models** are code, that require **data**, and are generated through **training**... 

### So how do you train a model? 

#### General Approaches in Machine Learning:
Certain pattern-recognition problems lend themselves to particular training techniques. Machine learning problems tend to fall into 1 of 3 camps: Supervised, Unsupervised, or Reinforcement Learning. When you are **training** your model, it is said to be **learning**.

 - **Supervised Learning:**  
	 - *Problem*: Often tries to do something humans already do well, or based on human expertise. 
	 - *Training*: Models are trained from labeled data (usually annotated by a human) and 'learn" a function that best predicts labels for new data. 
	 - *Applications*: Image Identification, Speech Recognition. 
	 - *Popular Algorithm Types*:
		 - Classification (predicting categories)
		 - Regression (predicting a value)
 - **Unsupervised Learning:** 
 	- *Problem*: Often helpful for providing insight into complex data problems which humans struggle to interpret. 
 	- *Training*: Models are trained to identify unknown patterns in data 
 	- *Applications*: Identifying trends in housing market or detecting anomalous credit card activity.
 	- *Popular Algorithm Types*:
		- Clustering (identifying groupings in data) 
		- Association (Identifying rules that associate data points) 
 - **Reinforcement Learning:** 
 	- *Problem*: Helpful for problems humans are not good at defining or problems with limited / state-dependent data.
 	- *Training*:  Model trained through agent interaction with a simulated or real environment - driven by a reward/risk mechanism to alter itself iteravly until it achieves its goal or maximizes its reward. 
 	- *Applications*:  robotics, self-driving cars, game playing AI, etc.
 	- *Popular Algorithm Types*:
		- Monte Carlo
		- Q-Learning
		- Deep Reinforcement Learning

Mind you, these are broad approaches to training models - there are many different sub-components to models, which can be trained via multiple methods and combined in unique ways to solve specific problems.

***Note:***
*Most of what we are discussing during this intensive falls under the category of "Supervised Learning" - simply because there are more opportunities for human input and exploration (data selection, labeling, parameters, subjective evaluation) without getting too into statistics (unsupervised learning) or simulation/hardware (reinforcement Learning). The latter two are equally worthly of study and exploration and there are resources and support for those methods as well.* 

## Terms that we've already seen, that we'll be seeing more of...
 - **Neural Networks**: A sequence of ML algorithms that are loosely modeled after the brain's neural network ([see 3B1B Video for a refresher](https://www.youtube.com/watch?v=aircAruvnKk))
 - **Deep Learning**: Refers to architectures with multiple/hidden layers (most NN models are Deep Neural Networks)
	 - Deep learning breaks down the feature recognition process and trains that too, allowing for more responsive and fine-tunable models (ex. the MINST NN example).
	- Deep Learning and Neural Networks have been incredibly popular over the past few years, with the increase in mass data and compute power - as they are very computationally intensive to train.
- **Transformers**: Deep Learning Models that leverage parallel computing (think lots of GPU/RAM) to maximize **Long-Short-Term Memory** - this allows for the model to make predictions based on a larger problem context. 
	- This makes transformers very popular for  **Natural Language Processing (NLP)** - where sentence and paragraph context can be factored into things like story generators, making them much more believable (ex. [AIDungeon](https://play.aidungeon.io/))
- **DIffusion Models**: Models trained to remove noise from images. These models were originally designed to solve problems of thermodynamic equilibrium, by"denoise-ing" imagery representing entropy of fluid systems. 
	- This process inadvertently made them good at generating images from random noise - which when guided by the addition of text or visual prompt - allows them to create much more creative 
- **Inference**: Is the process of running a model after it has been trained.  Inference is much less computationally intensive than training (in the case of extremely large models such as DALL-E, training is physically impossible on a consumer-grade computer). 
	- Though a model's architecture is set after it is trained, there are still a number of input parameters you can pass along with your data, during **inference** in order to control the final output. You will recall from Week 1 that [playgroundai](https://playgroundai.com/) allowed you to change the output dimensions, the prompt guidance, detail, and other parameters along with your image generation prompt.


### Further References
- [Google ML Glossary](https://developers.google.com/machine-learning/glossary) - clear concise definitions (with visuals) for a range of ML terms (if you come across a term you don't recognize, try looking it up here first)

# Engaging With Machine Learning Research

There are a number of ML driven applications that you already likely use or interact with- Google Translate, Automated Phone Systems, Ring Doorcams, etc. Though these tools use AI technology, you have no control or insight into the inference process.

Last week, when we explored [playgroundai](https://playgroundai.com/)  Text-to-Image generator, we were able to adjust some of the **inference** parameters for either the DALL-E or StableDiffusion  **Model** through a graphical user interface. These tools are fun, but there are limits to what we can do with them, and they still don't provide us with direct access to the **model**.
- What if we wanted to use a faster GPU to process the images? 
- What if we wanted to automate the image generation process and generate hundreds of images based on a list of prompts?
- What if we wanted integrate this model into a larger application or build our own interface?
- What if we want to combine this model with another model to create a novel workflow?

Machine Learning is a unique and fast-moving field, with novel models, approaches, papers being released daily. Luckily, many ML researchers release model code and datasets alongside their papers, so that you can experiment with it yourself. 

## Where do I find the research?
 - [Arxiv - Free Open Access Research Archive](https://arxiv.org/list/cs.AI/recent) - Papers released daily (if you see a ML paper elsewhere on the web, chances are it will redirect here) - Available in PDF, often with links to code. 
 - [Github](github.com) - Most research groups or individuals will have their paper's code hosted on github. If you find a paper or project that looks cool - check to see if they have a github, there may be additional resources or projects.
 - [Twitter](twitter.com) ... seriously, the ML research community is very active on Twitter. Impressive projects and papers are shared around quickly. If you start following ML researchers, or folks who engage with AI in a specific domain (business, Art, tech, gaming, policy, philosophy etc.) you will start to see ML discussions, demos, and tools regularly. 
 	- Here are a couple interesting folks to follow, to get you started:
 
## An AI Model in the Wild: Stable Diffusion
Stable Diffusion:
1. released
2. paper
3. github
4. huggingface/gradio
5. twitter:
	1. release announcement
	2. reduction in RAM requirements (proliferation of clones)
	3. Stand Alone Apps (PlaygroundAI)
	4. App integrations (ex. Photoshop, Blender, AR)
