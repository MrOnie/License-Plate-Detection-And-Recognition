# Libraries
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # remove warning message
#
import matplotlib.pyplot as plt
import numpy as np
#
from os.path import splitext
from keras.models import model_from_json
import cv2
from google.colab.patches import cv2_imshow
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.gridspec as gridspec
from importlib.machinery import SourceFileLoader
#
from sklearn.preprocessing import LabelEncoder
#
import pandas as pd
from IPython.display import clear_output
from IPython.display import Image as Im
#
from datetime import datetime, date, time # https://docs.python.org/3/library/datetime.html
import pytz
#
#https://stackoverflow.com/questions/51046454/how-can-we-use-selenium-webdriver-in-colab-research-google-com
from kora.selenium import wd
from time import sleep

notebooksPath = "/content/drive/MyDrive/LPDR/";

# ==================================================================== #
# Stage1. License Plate Detection
def stage1(path, img_name):
  vehicle, LpImg, cor = get_plate(path+img_name);
  plate_image = LpImg[0];
  plate = (255*plate_image).astype(np.uint8);
  plate = cv2.cvtColor(plate, cv2.COLOR_RGB2HSV);#BGR
  p_type = plate_type(plate, color_ranges, color_list);
  print("Plate type: "+ p_type)
  return vehicle, plate_image, p_type

def load_model(path, architecure, weights=''):
  try:
    path_arch = splitext(path+architecure)[0]
    with open('%s.json' % path_arch, 'r') as json_file:
      model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={})
    weights=architecure if weights == '' else weights
    path_weig = splitext(path+weights)[0]
    model.load_weights('%s.h5' % path_weig)
    return model
  except Exception as e:
    print(e)

# forward image through model and return plate's image and coordinates
# if error "No Licensese plate is founded!" pop up, try to adjust Dmin
def get_plate(image_path, Dmax=608, Dmin=350): #256 350
  vehicle = preprocess_image(image_path)
  ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
  side = int(ratio * Dmin)
  bound_dim = min(side, Dmax)
  _ , LpImg, _, cor = utilsWPOD.detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
  return vehicle, LpImg, cor

def preprocess_image(image_path):
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img / 255
  return img

def load_plate_types():
  color_list = ["comercial", "gubernamental", "gad", "particular"]
  color_ranges = {}
  color_ranges[color_list[0]+"Low"] = np.array([0, 170, 20], np.uint8)
  color_ranges[color_list[0]+"High"] = np.array([15, 255, 255], np.uint8)
  color_ranges[color_list[1]+"Low"] = np.array([16, 80, 20], np.uint8)
  color_ranges[color_list[1]+"High"] = np.array([33, 255, 255], np.uint8)
  color_ranges[color_list[2]+"Low"] = np.array([40, 170, 20], np.uint8)
  color_ranges[color_list[2]+"High"] = np.array([70, 255, 255], np.uint8)
  color_ranges[color_list[3]+"Low"] = np.array([0, 0, 150], np.uint8)
  color_ranges[color_list[3]+"High"] = np.array([178, 75, 255], np.uint8)
  return color_list, color_ranges

def plate_type(plate, types, color_list):
  """
  plate: HSV format & 0-255 range
  """
  portion_plate = plate[int(plate.shape[0]*0.1)-10:int(plate.shape[0]*0.1)+25, plate.shape[1]//2-50:plate.shape[1]//2+50, :]
  mask = []
  for c in color_list:
    m = cv2.inRange(portion_plate, types[c+"Low"], types[c+"High"])
    mask.append(np.sum(m/255))
  return color_list[mask.index(max(mask))]

# ==================================================================== #
# Stage2. License Plate Recognition
def stage2(plate_image, day):
  pl_image, gray, blur, binary, thre_mor = morph(plate_image, day)
  test_roi, crop_characters = characters(binary, thre_mor, pl_image)
  if len(crop_characters) == 0:
    test_roi, crop_characters = characters(remove_edges(binary), thre_mor, pl_image)
  print("{} chars detected".format(len(crop_characters)))
  final_string = ''
  for i, character in enumerate(crop_characters):
      c = np.array2string(predict_from_model(character,model_MNCR,labels))
      final_string+=c.strip("'[]")
  final_string = string_rectifier(final_string);
  print("Recognized plate: "+ final_string)  
  return thre_mor, test_roi, crop_characters, final_string

def morph(LpImg, day):
  # Scales, calculates absolute values, and converts the result to 8-bit.
  plate_image = cv2.convertScaleAbs(LpImg, alpha=(255.0))

  # convert to grayscale and blur the image
  gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) #COLOR_BGR2GRAY
  blur = cv2.GaussianBlur(gray,(7,7),0)

  # Applied inversed thresh_binary
  if day:
    binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  else:
    binary = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)

  kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #3, 5
  thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
  return plate_image, gray, blur, binary, thre_mor

# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

def characters(binary, thre_mor, plate_image):
  cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #binary

  # creat a copy version "test_roi" of plat_image to draw bounding box
  test_roi = plate_image.copy()

  # Initialize a list which will be used to append charater image
  crop_characters = []

  # define standard width and height of character
  digit_w, digit_h = 30, 60 # it may help to improve ***

  for c in sort_contours(cont):
      (x, y, w, h) = cv2.boundingRect(c)
      ratio = h/w
      #print(ratio)
      if 0.82<=ratio<=5: # Only select contour with defined ratio
          if 0.35<=h/plate_image.shape[0]<0.8 and w/plate_image.shape[1]<1/6:
              #print(ratio)
              # Draw bounding box arroung digit number
              cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2) #2

              # Sperate number and gibe prediction
              curr_num = thre_mor[y:y+h,x:x+w]
              curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
              _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
              crop_characters.append(curr_num)

  return test_roi, crop_characters

def remove_edges(binary):
  mask = np.zeros(binary.shape, dtype=np.uint8)
  cnts = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      area = cv2.contourArea(c)
      if area < 10000:
          cv2.drawContours(mask, [c], -1, (255,255,255), -1)
  result = cv2.bitwise_and(binary,binary,mask=mask)
  return result

# pre-processing input images and pedict with model
def predict_from_model(image, model, labels):
  image = cv2.resize(image, (80,80))
  image = np.stack((image,)*3, axis=-1)
  prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
  return prediction

def char_rectification():
  char_rect = {'O':'0', 'D':'0', 'Q':'0', 'I':'1', 'F': '1', 'J':'1', 'L':'1', 'B':'3', 'A':'4', 'H':'4', 'G':'6', 'T':'7', 'B':'8', 
  '0':'O', '1':'I', '2':'2', '3':'B', '4':'A', '6':'G', '7':'T', '8':'B'}
  return char_rect

def string_rectifier(full_string):
  if 6 <= len(full_string) <=7:
    full_string_r = ''
    for i in range(len(full_string)):
      if (full_string[i].isnumeric() and i < 3) or (full_string[i].isalpha() and i > 2):
        try:
          c = char_rect[full_string[i]]
        except:
          c = full_string[i]
      else:
        c = full_string[i]
      full_string_r += c;
  else:
    full_string_r = full_string;
  return full_string_r

# ==================================================================== #
# Printing
def print_plate(vehicle, LpImg, thre_mor, test_roi, final_string):
  fig = plt.figure(figsize=(12,6))
  grid = gridspec.GridSpec(ncols=2,nrows=2,figure=fig)
  fig.add_subplot(grid[0])
  plt.axis(False)
  plt.imshow(vehicle)
  fig.add_subplot(grid[1])
  plt.axis(False)
  plt.imshow(LpImg)
  fig.add_subplot(grid[2])
  plt.axis(False)
  plt.imshow(thre_mor, cmap='gray')
  fig.add_subplot(grid[3])
  plt.axis(False)
  plt.imshow(test_roi)
  plt.title(final_string)
  return

def is_printed(vehicle, vehicle_plate, thre_mor, test_roi, final_string, img_print = True):
  if img_print:
    print_plate(vehicle, vehicle_plate, thre_mor, test_roi, final_string)
  return

# ==================================================================== #
# Stage Vehicular traffic restriction
def restriction_schedule(res_type):
  sc_h = {}
  if res_type == "Pico y Placa":
    sc_d = {0: ['1', '2'], # Monday 0, plates 1 & 2
            1: ['3', '4'], # Tuesday 1
            2: ['5', '6'], # Wednesday 2
            3: ['7', '8'], # Thursday 3
            4: ['9', '0'], # Fryday 4
            5: [''],
            6: ['']}
    sc_h["bt1"] = time(7, 0, 0) # 7:00am
    sc_h["hi1"] = time(9, 30, 0) # 9:30am
    sc_h["bt2"] = time(16, 0, 0) # 16:00pm
    sc_h["hi2"] = time(19, 30, 0) # 19:30pm  
  elif res_type == "No circula":
    sc_d = {0: ['0', '1', '2', '3'], # Monday 0, plates 0 & 1 & 2 & 3
            1: ['2', '3', '4', '5'], # Tuesday 1
            2: ['4', '5', '6', '7'], # Wednesday 2
            3: ['6', '7', '8', '9'], # Thursday 3
            4: ['8', '9', '0', '1'], # Fryday 4
            5: [''],
            6: ['']}
    sc_h["bt1"] = time(7, 0, 0) # 7:00pm
    sc_h["hi1"] = time(19, 0, 0) # 19:00pm
  return sc_d, sc_h

def in_range(start, end, x):
  """Return true if x is in the range [start, end]"""
  if start <= end:
    return start <= x <= end
  else:
    return start <= x or x <= end

def ismember(A, B): #A whole, B part
  return np.sum([ np.sum(a == B) for a in A ])

def vehicular_traffic_restriction(res_type, date_time, verified_plate):
  sc_d, sc_h = restriction_schedule(res_type)
  if date_time == "current":
    tz_UIO = pytz.timezone('America/Guayaquil') 
    dt = datetime.now(tz_UIO)
  else:
    dt = datetime.strptime(date_time, '%y/%m/%d %H:%M:%S')
  c_day = date(dt.year, dt.month, dt.day)
  c_time = time(dt.hour, dt.minute, dt.second)
  restr = "CAN"
  if ismember(sc_d[c_day.weekday()], verified_plate[-1]):
    restricted = []
    for i in range(len(sc_h)//2):
      restricted.append(in_range(sc_h["bt"+str(i+1)], sc_h["hi"+str(i+1)], c_time))
    if sum(restricted):
      restr = "CANNOT"
  print('The car with plate number %s %s be on the road,\nthe day %s at %s,\naccording to "%s" schedule.'
          %(verified_plate, restr, c_day, c_time, res_type))
  #print('Restricted plates: ',sc_d[c_day.weekday()])
  return restr, c_day, c_time

def information(plate, place):
  if place == "DB":
    information_db = df[df["Placa"]==plate].T
    with open(notebooksPath+'DB/db_img.png', "rb") as db_img:
      ant_info = db_img.read()
    if information_db.empty:
      print("Not listed in DB")
    else:
      print(information_db[2::])
  elif place == "ANT":
    wd.get("https://sistematransito.ant.gob.ec:5038/PortalWEB/paginas/clientes/clp_criterio_consulta.jsp")
    #https://www.youtube.com/watch?v=lvFAuUcowT4&ab_channel=AaronJack
    #https://stackoverflow.com/questions/7867537/how-to-select-a-drop-down-menu-value-with-selenium-using-python
    id_type = wd.find_element_by_xpath("//select[@id='ps_tipo_identificacion']/option[4]")
    id_type.click()
    value = wd.find_element_by_xpath('//*[@id="ps_identificacion"]')
    value.send_keys(plate)
    search = wd.find_element_by_xpath('//*[@id="frm_consulta"]/div/a/img')
    search.click()
    sleep(2)
    wd.set_window_size(1050, 60)
    ant_info = wd.get_screenshot_as_png()
    Im(ant_info)
  return wd, ant_info

# ==================================================================== #
# Main Loading
def main():
  print("Loading WPOD...")
  # WPOD model
  wpod_net = load_model(notebooksPath, "wpod-net");
  # WPOD utils
  utilsWPOD = SourceFileLoader("utilsWPOD", notebooksPath+"utilsWPOD.py").load_module();
  print("Loaded WPOD!")
  # Plate types
  color_list, color_ranges = load_plate_types();
  print("Loading MobileNets...")
  # MobileNets Char. Recog. model + License Char. Recog. weights
  model_MNCR = load_model(notebooksPath, "MobileNets_character_recognition", "License_character_recognition_weight");
  # Labels
  labels = LabelEncoder()
  labels.classes_ = np.load(notebooksPath+'license_character_classes.npy')
  print("Loaded MobileNets!")
  # Char rectification
  char_rect = char_rectification()
  # dataframe of DB
  df = pd.read_excel (r"/content/drive/MyDrive/LPDR/DB/Placas.xlsx")
  return wpod_net, utilsWPOD, color_list, color_ranges, model_MNCR, labels, char_rect, df

wpod_net, utilsWPOD, color_list, color_ranges, model_MNCR, labels, char_rect, df = main()