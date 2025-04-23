# Media Informatics Assignment

Phase 1: Load and Reveal the Image
● Open the file (secret_image.csv).
● Figure out how to read the pixel data into Python using NumPy.
● After displaying the grayscale version, you must visualize the image using at least two
different colormaps (hot, cool, viridis).
(Colormaps are color filters that map pixel values to colors, helping to reveal hidden patterns.)
● Save and include all resulting visualizations in your submission.

Phase 2: Pattern Detection and Analysis
● Count how many black pixels exist in the image.
● Extract and print the coordinates of all black pixels.
● Determine the bounding box (min and max x and y values) that contains all the black pixels.
● Analyze the pattern: check if there are dark spots (black pixels) at some positions and
report whether there is a recognizable structure. (hint: look for symmetry or facial
features).

Phase 3: Modify the Image
● Convert the grayscale image into an RGB image.
● Change the color of the detected "eyes" to red.
● Add a blue border around the image.
● Flip the mouth upside down by creating a "sad face".
● Save and Display the modified image.

Phase 4: Apply a Noise Reduction Filter
● Apply a mean (average) filter to the final modified image in order to reduce the soft background noise while keeping the structure of the main pattern.
● Compare the filtered image with the original (noisy) version by displaying both side by side.
● Save and include both the noisy and the denoised images in your submission.

Phase 5: Please answer the following questions at the end of your google colab notebook:
● How many black pixels were found?
● What are the coordinates of the black pixels?
● What is the bounding box?
● What features did you detect in the image?
