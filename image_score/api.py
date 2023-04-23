# import dependencies
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve1d
import face_recognition
from itertools import compress
from brisque import BRISQUE
   

# 0. General Use. Functions to load, transform, and display images.

def load_image(img, mode='RGB'):
    """
    Loads an image as an numpy array. Most file formats are supported.
    
    args:
        img (filepath or array)- image,
        mode (str)- either RGB (color) or L (grayscale)
    return (array):
        image in file
    """
    if type(img)==str:
        return face_recognition.load_image_file(img, mode)
    else:
        if mode=='L':
            im = Image.fromarray(img) # make PIL image
            im = im.convert("L") # convert to grayscale
            return np.array(im)
        else:
            return img

def make_grayscale(img):
    """
    Takes an image and turns it grayscale.

    args:
        img (filepath or array)- image
    return (array):
        img with grayscale color values
    """
    return load_image(img, 'L')

def crop(image, location: tuple) -> np.ndarray:
    """
    Takes an image and a location coordinates tuple and gives image array for that section.

    args:
        image (filepath or array)- filepath or array for image,
        location (tuple)- four coordinates within image
    return (array):
        image cropped to the given location
    """
    img = load_image(image)
    top, right, bottom, left = location
    return img[top:bottom, left:right]

def display_image(image):
    """
    Takes either an image filepath or a numpy array of image and displays it.

    args:
        image (filepath or array)- image
    return (None):
        None
    """
    img = load_image(image)
    img = Image.fromarray(img) # convert to PIL image
    img.show() # show
    return

def display_faces(image, locations: list, color="white", width=3):
    """
    Takes a numpy array of image and an list of location tuples of faces on image,
    then displays the image with the faces marked with rectangles.
    
    args:
        image (filepath or array)- image,
        locations (list)- list of tuples of (4) coordinates within the image where faces are located,
        color (str)- color for boxes in display (ex: white, red, green),
        width (int)- width of boxes in display
    return (None):
        None
    """
    img = load_image(image)
    pil_image = Image.fromarray(img) # get PIL image
    pil_draw = ImageDraw.Draw(pil_image) # make it an ImageDraw object
    for loc in locations:
        top, right, bottom, left = loc # reorganize coordinates
        pil_draw.rectangle((left, bottom, right, top), outline =color, width=width) # draw rectangles
    pil_image.show() # display
    return

def display_section(image, location: tuple):
    """
    Takes image array and a set of coordinates to display a cropped section of the original image.

    args:
        img (filepath or array)- image,
        location (tuple)- four integers indicating (top, right, bottom, left) coordinates within image
    """
    img = load_image(image)
    top, right, bottom, left = location
    section = img[top:bottom, left:right]
    section = Image.fromarray(section)
    section.show()
    return



# 1. Blurriness. This section follows the algorithm proposed by Crété-Roffe et al. in
##   "The Blur Effect: Perception and Estimation with a New No-Reference Perceptual Blur Metric."

def _luminance(image) -> np.ndarray:
    """
    Takes either an image filepath or a numpy array of image and gets luminance component.

    args:
        image (filepath or array)- image
    return (array):
        luminance component. values scaled from 0 to 1.
        to display, multiply by 256 and call display_image
    """
    # load image as grayscale
    img = load_image(image, 'L')
    return img/256.0

def _blur_axis(L: np.ndarray, axis: int):
    """
    Takes luminance component and axis and blurs image alongside that axis.

    args:
        L (array)- luminance component,
        axis (int)- 0 (vertical) or 1 (horizontal)
    return (array):
        blurred image with same dimensions as L. 
        to display, multiply by 256 and call display_image
    """
    return convolve1d(L, [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9], axis=axis) # blur

def _abs_diff(M: np.ndarray, axis: int):
    """
    Takes any matrix and calculates its absolute difference along an axis.

    args:
        M (array)- matrix representing luminance component, blurred image, or pixel variation,
        axis (int)- 0 (vertical) or 1 (horizontal)
    return (array):
        difference matrix
    """
    m = len(M) # height of matrix
    n = len(M[0]) # width of matrix
    D = np.zeros((m,n)) # difference matrix to return
    if axis==0:
        for i in range(1,m):
            for j in range(0,n):
                D[i,j] = abs(M[i,j] - M[i-1,j])
    elif axis==1:
        for j in range(1,n):
            for i in range(0,m):
                D[i,j] = abs(M[i,j] - M[i,j-1])
    return D

def _blur_variation(D_L: np.ndarray, D_B: np.ndarray, axis: int):
    """
    Takes difference matrices of luminosity component and blurred luminosity component of the same image, 
    all differenced and/or blurred along axis 'axis,' and finds the variation between them.

    args:
        D_L- absolute difference matrix taken along axis 'axis' of luminosity component,
        D_B- absolute difference matrix taken along axis 'axis' of luminosity component blurred along axis 'axis',
        axis- 0 (vertical) or 1 (horizontal)
    return (array):
        variation matrix
    """
    
    m = len(D_L) # height of matrix
    n = len(D_L[0]) # width of matrix
    V = np.zeros((m,n)) # variation matrix to return
    if axis==0:
        for i in range(1,m):
            for j in range(1,n):
                V[i,j] = max(0, D_L[i,j]-D_B[i,j])
    elif axis==1:
        for j in range(1,n):
            for i in range(1,m):
                V[i,j] = max(0, D_L[i,j]-D_B[i,j])
    return V

def calculate_blur(img) -> float:
    """
    Calculates blurriness of an image from a scale to 0 (completely clear) to 1 (completely blurry).

    args:
        img (filepath or array)- filepath to or numpy array of an image
    return (float):
        blurriness intensity, from 0 (best) to 1 (worst)
    """
    # get original luminance component
    L = _luminance(img)
    # get blur along both axes
    B_ver = _blur_axis(L, 0)
    B_hor = _blur_axis(L, 1)
    # get abs diff images
    D_F_ver = _abs_diff(L, 0)
    D_F_hor = _abs_diff(L, 1)
    D_B_ver = _abs_diff(B_ver, 0)
    D_B_hor = _abs_diff(B_hor, 1)
    # get vertical and horizontal variations
    V_ver = _blur_variation(D_F_ver, D_B_ver, 0)
    V_hor = _blur_variation(D_F_hor, D_B_hor, 1)
    # get difference matrices of variation matrices
    D_V_ver = _abs_diff(V_ver, 0)
    D_V_hor = _abs_diff(V_hor, 0)
    # get sum of coefficients for abs diff matrices
    s_F_ver = np.sum(D_F_ver)
    s_F_hor = np.sum(D_F_hor)
    s_V_ver = np.sum(D_V_ver)
    s_V_hor = np.sum(D_V_hor)
    # get normalized vertical and horizontal blur scores
    b_F_ver = (s_F_ver-s_V_ver)/s_F_ver
    b_F_hor = (s_F_hor-s_V_hor)/s_F_hor
    # return max of both
    return max(b_F_ver, b_F_hor)

def blur_score(original, generated, report=False) -> float:
    """
    Takes two images (one of a real image, one for a SD-generated one)
    and calculates a score for the generated image in terms of blurriness.
    The score should lie from -0.3 to 1.3 and is equivalent to
    calculate_blur(generated) - 0.3*[calculate_blur(generated) - calculate_blur(original)].
    It can also print a report of the blurriness of both photos.

    args:
        original (filepath or array)- image taken by a camera,
        generated (filepath or array)- image generated by stable diffusion,
        report (bool)- optional, prints the individual blur scores for each image

    return (float):
        the blur score, from -0.3 (best) to 1.3 (worst)
    """
    og_blur = calculate_blur(original)
    gen_blur = calculate_blur(generated)
    if report==True:
        print(f'The original image has blur {round(og_blur, 3)} and the generated has blur {round(gen_blur, 3)}.')
    return gen_blur - 0.3*(og_blur - gen_blur)

# 2. Face Faithfulness. Measures if the number of faces in two images is the same,
#    then measures how similar the faces look to each other.

def _face_size(location: tuple) -> int:
    """
    Takes a face location coordinates tuple and calculates size of face.

    args:
        location (tuple): coordinates for face location
    return (int):
        face size in square pixels
    """
    top, right, bottom, left = location
    return (bottom-top)*(right-left)

def find_faces(image, model="hog"):
    """
    Takes an image array and finds coordinates for all objects reasonably recognized as a face.
    "cnn" strongly recommended for model if GPU acceleration is available.

    args:
        image (filepath or array)- image,
        model (str)- optional, default is hog (faster) but also has option cnn (more accurate, requires GPU acceleration)
    return (list):
        coordinate tuples for face locations
    """
    img = load_image(image)

    # get raw locations
    raw_locations = face_recognition.face_locations(img, 1, model)
    
    if len(raw_locations) > 1:
        # remove errors and background faces by looking at size
        face_sizes = list(map(_face_size, raw_locations))
        
        # remove faces <10% the size of the largest face
        condition = face_sizes > np.max(face_sizes)/10.0
        locations = list(compress(raw_locations, condition))
        return locations
    else:
        return raw_locations

def _get_encodings(img: np.ndarray, locs=[], model="hog") -> list:
    """
    Takes an image array and an optional list of face location tuples and 
    gets the face_recognition face encodings for each.

    args:
        img (array)- image,
        locs (list)- optional list of face location coordinate tuples
    return (list):
        arrays showing the 128-dimensional face encoding for each face at the locations specified.
    """
    if len(locs)==0:
        locations = find_faces(img, model)
    else:
        locations = locs
    if len(locations)==0:
        return []
    
    return face_recognition.face_encodings(img, locations)

def find_closest_face(og, location: tuple, 
                      gen, locs=[], 
                      model="hog", display=False) -> tuple:
    """
    Takes a face in an original image and a generated image and
    finds the location of the face in the comparison image most similar
    to the first encoding. Used mostly to check one's work.
    "cnn" strongly recommended for model if GPU acceleration is available.

    args:
        og (filepath or array)- image,
        location (tuple)- location of a face in 'img' that we want to compare to those in 'comparison',
        gen (filepath or array)- image containing faces that may be similar to encoding,
        locs (list)- optional list of locations for faces in 'generated',
        model (str)- optional, default is hog (faster) but also has option cnn (more accurate, requires GPU acceleration)
        display (bool)- prints the original face image, then the generated image with the closest face highlighted
    return (tuple):
        location coordinates for the comparison image showing the face closest to 'face',
        None if no faces are found in the comparison image
    """
    original = load_image(og)
    generated = load_image(gen)

    # get encoding for the first face
    encoding1 = _get_encodings(original, [location], model)[0]
    # get encodings for the comparison image
    if len(locs)==[]:
        locations = find_faces(generated, model)
    else:
        locations = locs
    if len(locations) < 1:
        return ()
    encodings2 = _get_encodings(generated, locations, model)
    # compare distances
    distances = face_recognition.face_distance(encodings2, encoding1)
    closest = locations[np.argmin(distances)]
    # "print"
    if display==True:
        display_section(original, location)
        display_faces(generated, [closest])
    # return location corresponding to the shortest distance
    return closest

def face_score(og, gen, model="hog", report=False) -> float:
    """
    Takes an original image and a comparison image and scores how dissimilar the faces in the generated
    are to the original. Score is calculated as
    0.5*(abs(pct change in number of faces)) + 0.5*(median face distance score for each face's best match)
    "cnn" strongly recommended for model if GPU acceleration is available.
    
    args:
        og (filepath or array)- image,
        gen (filepath or array)- stable diffusion-generated image
        model (str)- optional, default is hog (faster) but also has option cnn (more accurate, requires GPU acceleration)
        report (bool)- prints the number of faces in each image and the lowest face distance match for each face
    return (float):
        face inaccuracy intensity, from 0 (best) to 1 (worst)
    """
    original = load_image(og)
    generated = load_image(gen)

    # get locations and numbers of all faces
    og_locations = find_faces(og, model) # find_faces only works with string inputs. why?
    gen_locations = find_faces(gen, model)
    og_nfaces = len(og_locations)
    gen_nfaces = len(gen_locations)
    if report==True:
            print(f'The original image had {og_nfaces} faces and the generated had {gen_nfaces}.')
    
    if og_nfaces!=0 and gen_nfaces!=0: # only check face faithfulness if both images have faces
        # get all face encodings
        og_encodings = _get_encodings(original, og_locations, model)
        gen_encodings = _get_encodings(generated, gen_locations, model)
        # find closest face distance for each original face
        best_distances = []
        best_locations = []
        for encoding in og_encodings:
            distances = face_recognition.face_distance(gen_encodings, encoding)
            best_distances.append(min(distances))
            best_locations.append(gen_locations[np.argmin(distances)])
        if report==True:
            # print list of all original face locations and the locations/distances of their closest match
            # can be referenced for high-level checking
            for i in range(len(best_distances)):
                print(f'original: {og_locations[i]}\tclosest: {best_locations[i]}\tdistance: {round(best_distances[i], 3)}')
        # calculate score
        pctchng_nfaces = abs(og_nfaces - gen_nfaces)/og_nfaces
        med_distance = np.median(best_distances)
        return 0.5*pctchng_nfaces + 0.5*med_distance
    elif og_nfaces==0 and gen_nfaces==0: # no face faithfulness to calculate
        return 0.0
    else: # generator either removed all faces or created new ones when none existed
        return 1.0

# 3. Composition. Check each face or face group's placement within the photograph.
##   Vertically, eyes should be 1/3, 1/2, or 2/3 of the way down from the top of the photo.
##   Horizontally, each face or face group should be near 1/3, 1/2, or 2/3 from the left of the photo.

def _find_composition_points(image):
    """
    Takes an image and returns a list in the form
    [[1/3 height, 1/2 height, 2/3 height], [1/3 width, 1/2 width, 2/3 width]]

    args:
        image (filepath or array): image
    return (list):
        lists for composition heights and widths of interest
    """
    img = load_image(image)
    
    height = len(img)
    width = len(img[0])

    heights = [height/3, height/2, 2*height/3]
    widths = [width/3, width/2, 2*width/3]
    return heights, widths

def _get_eye_heights(img: np.ndarray, locations: list) -> list:
    """
    Takes an image and a list of face locations and checks how far the eye heights are
    from 1/3 of the way from the top of the photo, 1/3 of the way from the bottom, or the center.
    Returns median vertical distance of eyes from goal.

    args:
        img (array): image
        locations (list): tuples of face locations
    return (list):
        heights of eyes for each face in 'locations'
    """
    landmarks = face_recognition.face_landmarks(img, locations, "small") # get face landmarks
    heights = []
    for dict in landmarks:
        left_height = (dict['left_eye'][0][1] + dict['left_eye'][1][1])/2
        right_height = (dict['right_eye'][0][1] + dict['right_eye'][1][1])/2
        heights.append((left_height+right_height)/2) # get average height of both eyes of person
    return heights

def _vertical_score(img: np.ndarray, locations: list) -> float:
    """
    Takes an image and a list of face locations in the photo and checks how far the eye heights are
    from 1/3, 1/2, or 2/3 of the way from the top of the photo.
    Returns intensity of vertical distance from the desired height the eyes are farthest from.
    Intensity is scored by seeing which goal height the eyes as a whole are closest to,
    then taking thrice the median vertical distance from that goal.

    args:
        img (array): image
        locations (list): tuples of face locations
    return (float):
        3*(median vertical distance from closest goal)/(height of image)
    """

    heights = _get_eye_heights(img, locations) # get eye heights
    if len(heights)==0: # no eyes were found, but we know the face exists
        for location in locations:
            height = location[0] + (location[2] - location[0])/3
            heights.append(height)
    goals = _find_composition_points(img)[0]
    median_distances = [] # list of the median distance of the eyes from each goal
    for goal in goals:
        distances_from_goal = list(map(lambda h: abs(goal-h), heights)) # get each eye's distance from goal
        median_distances.append(np.median(distances_from_goal))
    closest_index = np.argmin(median_distances)
    
    # return score
    return 3*median_distances[closest_index]/len(img)

def _group_faces(img: np.ndarray, location1: tuple, location2: tuple, border=0.2, scale=0.5) -> bool:
    """
    Takes an image and two face locations and decides whether or not to group them together.
    Faces are put in the same group when they are within a certain distance apart
    and are within a certain proportion of the same size as each other.

    args:
        img (array)- image
        location1, location2 (tuple)- coordinate tuple for each face
        border (float)- optional, 0 to 1. closeness that images must be as a proportion of the full image height/width for grouping
        scale (float)- optional, 0 to 1. percentage of the larger image a smaller image must be for grouping
    return (bool):
        whether faces should be grouped
    """
    def _expand_rectangle(rect: tuple, img=img, border=border) -> tuple:
        """
        Takes a rectangle in (top, right, bottom, left) and expands its borders
        by border*(height or width of entire image).
        """
        top, right, bottom, left = rect
        height = len(img)
        width = len(img[0])
        height_change = int(height * border)
        width_change = int(width * border)
        # get new coords
        top = max(0, top-height_change)
        right = min(width-1, right+width_change)
        bottom = min(height-1, bottom+height_change)
        left = max(0, left-width_change)
        return (top, right, bottom, left)

    def _rectangle_overlap(rect1: tuple, rect2: tuple) -> bool:
        """
        Checks if two rectangles in (top, right, bottom, left) form overlap.
        """
        top1, right1, bottom1, left1 = rect1
        top2, right2, bottom2, left2 = rect2
        # separating axis theorem
        condition1 = (right1 < left2) | (right2 < left1) # proves no horizontal overlap
        condition2 = (top1 > bottom2) | (top2 > bottom1) # proves no vertical overlap
        return not(condition1 | condition2)
    
    def _size_similarity(rect1: tuple, rect2: tuple, scale=scale) -> bool:
        """
        Checks if two rectangles in (top, right, bottom, left) form are within 'scale' area of each other.
        """
        size1 = _face_size(rect1)
        size2 = _face_size(rect2)
        if size1==0 | size2==0:
            return False
        if size1 <= size2:
            return size1/size2 >= scale
        else:
            return size2/size1 >= scale
    
    expanded1 = _expand_rectangle(location1)
    expanded2 = _expand_rectangle(location2)
    return _rectangle_overlap(expanded1, expanded2) & _size_similarity(location1, location2)

def get_face_groups(image, model="hog", locs=[], border=0.2, scale=0.5) -> list:
    """
    Takes an image and a list of face locations and returns a list of locations of face groups.
    "cnn" strongly recommended for model if GPU acceleration is available.

    args:
        image (filepath or array)- image
        locs (list)- coordinate tuples of face locations
        model (str)- optional, default is hog (faster) but also has option cnn (more accurate, requires GPU acceleration)
        border (float)- optional, how close faces need to be to each other to be grouped
        scale (float)- optional, how similar in size faces need to be to be grouped
    return (list):
        tuples of the locations of each face group
    """
    img = load_image(image)

    if len(locs)==0:
        locations = find_faces(img, model)
    else:
        locations = locs

    if len(locations)==0:
        return []

    nfaces = len(locations)
    
    # create face group adjacency matrix
    face_adjacency = np.zeros((nfaces, nfaces))
    for i in range(nfaces):
        for j in range(i+1, nfaces):
            if _group_faces(img, locations[i], locations[j], border, scale)==True:
                face_adjacency[i,j]=1
    
    # transform adjacency matrix to list of groups using breadth-first search
    groups = []
    remaining_nodes = list(range(nfaces))
    # BFS
    def _bfs(node):
        visited = []
        queue = []
        visited.append(node)
        queue.append(node)
        while queue:
            m = queue.pop(0)
            for j in range(len(face_adjacency)):
                other = face_adjacency[m,j]
                if (other==1) and (j not in visited):
                    visited.append(j)
                    if j in remaining_nodes:
                        remaining_nodes.remove(j)
                    queue.append(j)
        return visited
    # implement
    while remaining_nodes:
        groups.append(_bfs(remaining_nodes.pop(0)))

    group_locs = []
    for group in groups:
        locs = [locations[i] for i in group] # narrow to locations in group
        # get bounds of box that contains all faces
        top = min(list(map(lambda a: a[0], locs)))
        right = max(list(map(lambda a: a[1], locs)))
        bottom = max(list(map(lambda a: a[2], locs)))
        left = min(list(map(lambda a: a[3], locs)))
        group_locs.append((top, right, bottom, left))
    
    return group_locs

def _get_face_centers(img: np.ndarray, group_locs: list) -> list:
    """
    Takes an image and a list of face locations and gets the horizontal middles for each group.

    args:
        img (array)- image
        group_locs (list)- coordinate tuples of rectangles encompassing full face groups
    return (list):
        floats of the width-wise center of each face group
    """
    
    # find horizonal centers of each group
    centers = []
    for group in group_locs:
        right = group[1]
        left = group[3]
        # find horizontal center
        centers.append((right+left)/2)
    
    return centers

def _horizontal_score(img: np.ndarray, locations: list, model="hog", border=0.2, scale=0.5) -> float:
    """
    Takes an image and a list of the face group's horizontal locations and checks how far each are from
    1/3, 1/2, or 2/3 from the left of the photo.
    Intensity is scored by seeing which goal width each individual face group is closest to,
    taking the median, then multiplying by 3.
    "cnn" strongly recommended for model if GPU acceleration is available.

    args:
        img (array)- image
        locations (list)- coordinate tuples of each face location
        model (str)- optional, default is hog (faster) but also has option cnn (more accurate, requires GPU acceleration)
        border (float)- optional, 0 to 1. sensitivity for face distance to be grouped
        scale (float)- optional, 0 to 1. sensitivity for face size closeness to be grouped
    return (float):
        3*(median horizontal distance from closest goal)/(width of image)
    """

    goals = _find_composition_points(img)[1] # best number of pixels from the left for a group center to be
    groups = get_face_groups(img, model, locations, border, scale)
    centers = _get_face_centers(img, groups) # group faces and find the horizontal middles of each group

    # evaluate
    distances = []
    for center in centers:
        # save the distance from the closest goal width
        distances.append(min(abs(goals[0]-center), abs(goals[1]-center), abs(goals[2]-center)))
    
    # return score
    return 3*np.median(distances)/len(img[0])

def composition_score(gen, model="hog", border=0.2, scale=0.5, report=False) -> float:
    """
    Takes a generated image and scores how poor its composition is, aka the distance of its face groups
    from 'goal' areas according to the centering and 1/3 rules.

    args:
        gen (filepath or array)- image
        border (float)- optional, 0 to 1. sensitivity for face distance to be grouped
        scale (float)- optional, 0 to 1. sensitivity for face size closeness to be grouped
    return (float):
        composition distance from points of interest, from 0 (best) to 1 (worst)
    """

    # get all faces in image
    generated = load_image(gen)

    # get locations and numbers of all faces
    gen_locations = find_faces(generated, model)
    gen_nfaces = len(gen_locations)

    # we use the eye-level test only if there are <4 people in a photo
    if gen_nfaces==0:
        return 0.0 # no face location composition to evaluate
    if gen_nfaces < 4:
        vertical_score = _vertical_score(generated, gen_locations)
        horizontal_score = _horizontal_score(generated, gen_locations, model, border, scale)
        if report==True:
            print(f'vertical score={round(vertical_score, 3)} and horizontal score={round(horizontal_score, 3)}.')
        return (vertical_score + horizontal_score)/2
    else:
        horizontal_score = _horizontal_score(generated, gen_locations, model, border, scale)
        if report==True:
            print(f'horizontal score={round(horizontal_score, 3)}.')
        return horizontal_score

# 4. Naturalness. Scores each photo on levels of noise, oversmoothing, or other distortion.

def calculate_distortion(img) -> float:
    """
    Calculates distortion of an image from a scale to 0 (completely natural) to 1 (completely distorted).

    args:
        img (filepath or array)- filepath to or numpy array of an image
    return (float):
        distortion intensity, from 0 (best) to 1 (worst)
    """
    
    image = load_image(img)
    scorer = BRISQUE(url=False)
    score = scorer.score(image)/100

    return score

def distortion_score(og, gen, report=False) -> float:
    """
    Takes an original and generated image and scores how natural/undistorted the light distribution
    in the generated image looks, also compared to the original, using BRISQUE. The score is calculated
    as distortion(generated) - 0.3*(distortion(original) - distortion(generated)) and should lie between
    -0.3 and 1.3.

    args:
        og (filepath or array)- original image
        gen (filepath or array)- generated image
        report (bool)- optional, prints the BRISQUE scores for the original and generated image
    return (float):
        distortion of generated image and improvement from the original, from -0.3 (best) to 1.3 (worst)
    """

    og_score = calculate_distortion(og)
    gen_score = calculate_distortion(gen)

    if report==True:
        print(f'The original image has distortion {round(og_score, 3)} and the generated has distortion {round(gen_score, 3)}.')

    return gen_score - 0.3*(og_score - gen_score)


# 5. Total Scores.

def score(original, generated, model="hog", border=0.2, scale=0.5, report=False) -> dict:
    """
    Takes an original and generated image and scores the generated image on the metrics of blurriness,
    face faithfulness to original, photo composition, and distortion. It also shows the improvement
    on blur and distortion.
    "cnn" strongly recommended for model if GPU acceleration is available.

    args:
        original (filepath or array)- original image
        generated (filepath or array)- generated image
        model (str)- optional, default is hog (faster) but also has option cnn (more accurate, requires GPU acceleration)
        border (float)- optional, 0 to 1. sensitivity for face distance to be grouped
        scale (float)- optional, 0 to 1. sensitivity for face size closeness to be grouped
        report (bool)- optional, prints individual blur scores and face locations, matches, and distances
    return (dict):
        blurriness, face dissimilarity, and composition scores
    """

    # find scores
    og_blur = calculate_blur(original)
    gen_blur = calculate_blur(generated)
    if report==True:
        print(f'The original image has blur {round(og_blur, 3)} and the generated has blur {round(gen_blur, 3)}.')
    dissimilarity = face_score(original, generated, model, report)
    composition = composition_score(generated, model, border, scale, report)
    og_distortion = calculate_distortion(original)
    gen_distortion = calculate_distortion(generated)
    if report==True:
        print(f'The original image has distortion {round(og_distortion, 3)} and the generated has distortion {round(gen_distortion, 3)}.')

    return {"blurriness": gen_blur,
            "blur pct change": (gen_blur-og_blur)/og_blur,
            "face dissimilarity": dissimilarity,
            "compositional distance": composition,
            "distortion": gen_distortion,
            "distortion pct change": (gen_distortion-og_distortion)/og_distortion}
