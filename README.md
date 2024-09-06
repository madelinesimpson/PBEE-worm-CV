# PBEE-worm-CV

*This is not a funtioning app*

This is a central hub for code and resources used in the PBEE Worm CV project.

Since this is such a hodgepodge of random files, let me clarify what different things are.

The “acutally working” folder is the code needed to create the image that was shown yesterday (green = alive, yellow = cluster, red = dead). Classification_model.h5 is the trained neural network saved to a file. I trained the network on all of the labeled images in the “test” folder.

“wormAppUI” is a folder containing all of the code for that HTML web app I used as a demo on the call with Nic and Levi.

“rectangle_model.py” is where you can see the actual code for the neural network. I use the frameworks Tensorflow and Keras, both of which have a ton of documentation online. “rectangle_crop.py” in “actually working” is where you can find the code for how I turned worms into rectangles.

Here’s a link to the projects I used as references for writing this model:
https://keras.io/examples/vision/image_classification_from_scratch/
https://keras.io/examples/vision/mnist_convnet/

For the cluster problem that I’m working on right now, you can check out “individual_worm_data.py” (plotting circles on individual worms), “cluster_worm_data.py” (separating clusters into branches and intersections, and plotting circles on those), and “cluster_run.py” (the code to run the functions in the other two files. 

Here is the research paper I am basing that work off of:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3048333/
You can see what I mean by plotting circles along the worm in the images they show.
