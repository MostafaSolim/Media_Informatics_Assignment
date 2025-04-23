import numpy
import matplotlib.pyplot as plt
import cv2

thingy = numpy.genfromtxt('secret_image.csv', delimiter=',', dtype=numpy.float64)

print('Default')
default = plt.imshow(thingy)
plt.savefig('Default_Image.png')
plt.show()

print('GreyScale')
GreyIMG = plt.imshow(thingy, cmap='gray')
plt.savefig('Grey_Scale_Image.png')
plt.show()

print()
print('hot')
hotIMG = plt.imshow(thingy, cmap = 'hot')
plt.savefig('Hot_Image.png')
plt.show()

print()
print('cool')
coolIMG = plt.imshow(thingy, cmap = 'cool')
plt.savefig('Cool_Image.png')
plt.show()

# ● Count how many black pixels exist in the image
# 20x20
print("total amount of pixels", thingy.size)
print("Since it is a square, you can pull the square root of this number")
print()

# ● Extract and print the coordinates of all black pixels.
coords = numpy.argwhere(thingy == 0)
print('Here are the coordinates of the black pixels of the default image')
if(coords.size==0):
  print('this is empty btw')
else :
  print(coords)
print()
print('total amount of black pixels:' , coords.size)
print()

# ● Determine the bounding box (min and max x and y values) that contains all the black pixels.
maximumX = numpy.max(coords[:, 1])
minimumX = numpy.min(coords[:, 1])
maximumY = numpy.max(coords[:, 0])
minimumY = numpy.min(coords[:, 0])

print('=======================================')
print('               Bounding Box')
print('=======================================')

print('minimum x:' , minimumX)
print('minimum y:' , minimumY)
print('maximum x:' , maximumX)
print('maximum y:' , maximumY)
print()

# Check if there are dark spots (black pixels) at some positions
print('Width:', maximumX - minimumX)
print('Height:', maximumY - minimumY)
print('Ratio:', (maximumX - minimumX) / (maximumY - minimumY))

midX = (minimumX+maximumX)/2
midY = (minimumY+maximumY)/2

print('middle point X:' , midX)
print('middle point Y:' , midY)

# report whether there is a recognizable structure. (hint: look for symmetry or facial features)
left = numpy.sum(coords[:, 1] < midX)
right = numpy.sum(coords[:, 1] > midX)
print('=======================================')
print('               Symmetry')
print('=======================================')
print('Left:',left)
print('Right:',right)

#Phase 3
# Read the CSV file
data = numpy.genfromtxt('secret_image.csv', delimiter=',')

# Create RGB image
height, width = data.shape
rgb_image = numpy.stack([data, data, data], axis=-1).astype(numpy.uint8)

# Find black pixels (value 0)
black_pixels = numpy.argwhere(data == 0)

# Identify eyes (top 30% of image, symmetric points)
eye_candidates = []
for y, x in black_pixels:
    if y < height * 0.3:  # Only top 30% for eyes
        symmetric_x = width - 1 - x
        if (y, symmetric_x) in [(p[0], p[1]) for p in black_pixels]:
            eye_candidates.append((x, y))

# Mark eyes red (using two most central candidates)
if len(eye_candidates) >= 2:
    eye_candidates.sort(key=lambda c: abs(c[0] - width//2))  # Sort by centrality
    for x, y in eye_candidates[:2]:  # Take two most central
        rgb_image[y, x] = [255, 0, 0]
        rgb_image[y, width-1-x] = [255, 0, 0]  # Symmetric pair

# Add blue border
rgb_image[0, :] = [0, 0, 255]  # Top
rgb_image[-1, :] = [0, 0, 255]  # Bottom
rgb_image[:, 0] = [0, 0, 255]  # Left
rgb_image[:, -1] = [0, 0, 255]  # Right

# Specifically modify the mouth corner pixels
# First, identify mouth pixels
mouth_pixels = [(y, x) for y, x in black_pixels if y > height * 0.4]

# We'll need to find the horizontal line (main part of the mouth) and the corner pixels
horizontal_mouth_pixels = []
corner_pixels = []

if mouth_pixels:
    mouth_ys = [y for y, x in mouth_pixels]
    main_mouth_y = max(set(mouth_ys), key=mouth_ys.count)  # Most common y-value

    # Separate horizontal and corner pixels
    for y, x in mouth_pixels:
        if y == main_mouth_y:
            horizontal_mouth_pixels.append((y, x))
        else:
            corner_pixels.append((y, x))

    # Sort corners by x-coordinate
    corner_pixels.sort(key=lambda p: p[1])
    if len(corner_pixels) >= 2:
        left_corner = corner_pixels[0]
        right_corner = corner_pixels[-1]

        # Create background color (average of non-black pixels)
        bg_color = int(numpy.mean(data[data > 0]))

        # Remove original corner pixels
        rgb_image[left_corner[0], left_corner[1]] = [bg_color, bg_color, bg_color]
        rgb_image[right_corner[0], right_corner[1]] = [bg_color, bg_color, bg_color]

        # Calculate new position for corners (same x, but below the mouth line)
        new_y = main_mouth_y + (main_mouth_y - left_corner[0])  # Mirror around main_mouth_y

        # Add new corner pixels at the yellow dot positions
        rgb_image[new_y, left_corner[1]] = [0, 0, 0]
        rgb_image[new_y, right_corner[1]] = [0, 0, 0]

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(rgb_image)
plt.title('Modified Image (Sad Face)')
plt.axis('off')
plt.show()

# Save result
plt.imsave('sad_face_result.png', rgb_image)

# Phase 4
# ● Apply a mean (average) filter to the final modified image in order to reduce the soft background noise while keeping the structure of the main pattern.
rgb = rgb_image
filtered = cv2.blur(rgb, (3, 3))
plt.figure(figsize=(12, 6))

# Original
plt.subplot(1, 2, 1)
plt.imshow(rgb.astype(numpy.uint8))
# ● Save and include both the noisy and the denoised images in your submission.
plt.title('Original Modified Image')
#plt.savefig('Noisy_Image.png')
plt.axis('off')

# Filtered
plt.subplot(1, 2, 2)
plt.imshow(filtered.astype(numpy.uint8))
# ● Save and include both the noisy and the denoised images in your submission.
plt.title('Denoised Image (Mean Filter)')
#plt.savefig('Denoised_Image(Mean Filter).png')
plt.axis('off')

# ● Compare the filtered image with the original (noisy) version by displaying both side by side.
plt.tight_layout()
plt.savefig('Comparison_Noise_vs_Denoised.png')
plt.show()

cv2.imwrite('Noisy_Image.png', cv2.cvtColor(rgb.astype(numpy.uint8), cv2.COLOR_RGB2BGR))
cv2.imwrite('Denoised_Image.png', cv2.cvtColor(filtered.astype(numpy.uint8), cv2.COLOR_RGB2BGR))