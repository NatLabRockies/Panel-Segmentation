"""
Panel detection class
"""

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from matplotlib import cm
import requests
from PIL import Image
from os import path

panel_seg_model_path = path.join(path.dirname(__file__), 'VGG16Net_ConvTranpose_complete.h5')
panel_classification_model_path = path.join(path.dirname(__file__), 'VGG16_classification_model.h5')

class PanelDetection():
    '''
    A class for training a deep learning architecture, 
    detecting solar arrays from a satellite image, performing spectral
    clustering, and predicting the Azimuth.
    '''
    def __init__(self, model_file_path = './VGG16Net_ConvTranpose_complete.h5', 
                 classifier_file_path = './VGG16_classification_model.h5'):
        
        #This is the model used for detecting if there is a panel or not
        self.classifier = load_model(classifier_file_path, 
                                     custom_objects=None, 
                                     compile=False)
        
        self.model = load_model(model_file_path, 
                                custom_objects=None, 
                                compile=False)
        


    def hasPanels(self, test_data):
        """
        This function is used to predict if there is a panel in an image or not. 
        Note that it uses a saved classifier model we have trained and not the 
        segmentation model.       
        
        Parameters
        -----------
        test_data: (nparray float or int) 
            the satellite image. The shape should be [a,640,640,3] where 
            'a' is the number of data or (640,640,3) if it is a single image
                                       
        Returns
        -----------
        Boolean. Returns True if solar array is detected in an image, and False otherwise.
        """
        #Check that the input is correct
        if type(test_data) != np.ndarray:
            raise TypeError("Variable test_data must be of type Numpy ndarray.")
        #Test that the input array has 3 to 4 channels
        if (len(test_data.shape) > 4) | (len(test_data.shape) < 3):
            raise ValueError("Numpy array test_data shape should be 3 dimensions if a single image, or 4 dimensions if a batch of images.")        
        test_data = test_data/255
        #This ensures the first dimension is the number of test data to be predicted
        if test_data.ndim == 3:
            test_data = test_data[np.newaxis, :]
        prediction = self.classifier.predict(test_data)
        #index 0 is for no panels while index 1 is for panels
        if prediction[0][1] > prediction[0][0]:
            return True 
        else:
            return False
        

    def detectAzimuth(self, in_img, number_lines=5):
        """
        This function uses canny edge detection to first extract the edges of the input image. 
        To use this function, you have to first predict the mask of the test image 
        using testSingle function. Then use the cropPanels function to extract the solar 
        panels from the input image using the predicted mask. Hence the input image to this 
        function is the cropped image of solar panels.
        
        After edge detection, Hough transform is used to detect the most dominant lines in
        the input image and subsequently use that to predict the azimuth of a single image
  
        Parameters
        -----------
        in_img: (nparray uint8) 
            The image containing the extracted solar panels with other pixels zeroed off. Dimension is [1,640,640,3]
        number_lines: (int)  
            This variable tells the function the number of dominant lines it should examine.
            We currently inspect the top 10 lines.
            
        Returns
        -----------
        azimuth: (int) 
            The azimuth of the panel in the image.
        """
        #Check that the input variables are of the correct type
        if type(in_img) != np.ndarray:
            raise TypeError("Variable in_img must be of type Numpy ndarray.")
        if type(number_lines) != int:
            raise TypeError("Variable number_lines must be of type int.")
        #Run through the function
        edges = cv2.Canny(in_img[0],50,150,apertureSize=3)
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
        h, theta, d = hough_line(edges, theta=tested_angles)
        origin = np.array((0, edges.shape[1]))
        ind =0
        azimuth = 0
        az = np.zeros((number_lines))
        # Classic straight-line Hough transform
        # Set a precision of 0.5 degree.        
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=number_lines, threshold =0.25*np.max(h))):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
                
            deg_ang = int(np.rad2deg(angle))
            if deg_ang >= 0:
                az[ind] = 90+deg_ang
            else:
                az[ind] = 270 + deg_ang
            ind =ind+1
        unique_elements, counts_elements = np.unique(az, return_counts=True)
        check = counts_elements[np.argmax(counts_elements)]
        if check == 1:
            for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1, threshold =0.25*np.max(h))):
                deg_ang = int(np.rad2deg(angle))
                if deg_ang >= 0:
                    azimuth = 90+deg_ang
                else:
                    azimuth = 270 + deg_ang
        else:
            azimuth = (unique_elements[np.argmax(counts_elements)])
        return azimuth    

    
    def cropPanels(self, test_data, test_res):
        """
        This function basically isolates regions with solar panels in a 
        satellite image using the predicted mask. It zeros out other pixels that does not 
        contain a panel.
        You can use this for a single test data or multiple test data. 
        
        Parameters 
        ----------
        test_data:  (nparray float)
            This is the input test data. This can be a single image or multiple image. Hence the 
            dimension can be (640,640,3) or (a,640,640,3)
        test_res:   (nparray float) 
            This is the predicted mask of the test images passed as an input and used to crop out the 
            solar panels. dimension is (640,640)
        
        Returns 
        ----------
        new_test_res: (nparray uint8) 
            This returns images here the solar panels have been cropped out and the background zeroed. 
            It has the same shape as test data.  The dimension is [a,640,640,3] where a is the number of
            input images
            
        """
        #Check that the input variables are of the correct type
        if type(test_data) != np.ndarray:
            raise TypeError("Variable test_data must be of type Numpy ndarray.")
        if type(test_res) != np.ndarray:
            raise TypeError("Variable test_res must be of type Numpy ndarray.")            
        #Convert the test_data array from 3D to 4D
        if test_data.ndim == 3:
            test_data = test_data[np.newaxis, :]
        new_test_res = np.uint8(np.zeros((test_data.shape[0],640,640,3)))
        for ju in np.arange(test_data.shape[0]):
            try:
                in_img = test_res[ju].reshape(640,640)
            except:
                in_img = test_res.reshape(640,640)
            in_img[in_img < 0.9] = 0
            in_img[in_img >= 0.9] = 1
            in_img = np.uint8(in_img)
            test2 = np.copy(test_data[ju])
            test2[(1-in_img).astype(bool),0] = 0
            test2[(1-in_img).astype(bool),1] = 0
            test2[(1-in_img).astype(bool),2] = 0
            new_test_res[ju] = test2    
        return new_test_res
        
    
    def plotEdgeAz(self, test_results, no_lines=5, 
                    no_figs=1, save_img_file_path = None,
                    plot_show = False):
        """
        This function is used to generate plots of the image with its azimuth
        It can generate three figures or one. For three figures, that include the 
        input image, the hough transform space and the input images with detected lines.
        For single image, it only outputs the input image with detected lines.
        
        Parameters 
        ----------
        test_results: (nparray float64 or unit8) 
            8-bit input image. This variable represents the predicted images from the segmentation model. Hence the 
            dimension must be [a,b,c,d] where [a] is the number of images, [b,c] are the dimensions
            of the image - 640 x 640 in this case and [d] is 3 - RGB
        no_lines: (int) 
            default is 10. This variable tells the function the number of dominant lines it should examine.                  
        no_figs: (int) 
            1 or 3. If the number of figs is 1. It outputs the mask with Hough lines and the predicted azimuth
            However, if the number of lines is 3, it gives three plots. 
                1. The input image,
                2. Hough transform search space
                3. Unput image with houghlines and the predicted azimuth
                          
        save_img_file_path: (string) 
            You can pass as input the location to save the plots
        plot_show: Boolen: If False, it will supress the plot as an output and just save the  plots in a folder
        
        Returns 
        ----------
        Plot of the masked image, with detected Hough Lines and azimuth estimate.
        """
        #Check that the input variables are of the correct type
        if type(test_results) != np.ndarray:
            raise TypeError("Variable test_results must be of type Numpy ndarray.")
        if type(no_lines) != int:
            raise TypeError("Variable no_lines must be of type int.")  
        if type(no_figs) != int:
            raise TypeError("Variable no_figs must be of type int.")              
        if type(plot_show) != bool:
            raise TypeError("Variable no_figs must be of type boolean.")  
        
        for ii in np.arange(test_results.shape[0]):
            #This changes the float64 to uint8
            if (test_results.dtype is np.dtype(np.float64)):
                in_img = test_results[ii].reshape(640,640)
                in_img[in_img < 0.9] = 0
                in_img[in_img >= 0.9] = 1
                in_img = np.uint8(in_img)

            in_img = test_results[ii]
            #Edge detection
            edges = cv2.Canny(in_img,50,150,apertureSize=3)
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
            h, theta, d = hough_line(edges, theta=tested_angles)
            az = np.zeros((no_lines))
            origin = np.array((0, edges.shape[1]))
            ind =0
            # Generating figure 1            
            fig, ax = plt.subplots(1, no_figs, figsize=(10, 6))
            if no_figs == 1:
                ax.imshow(edges)# cmap=cm.gray)
                for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=no_lines, threshold =0.25*np.max(h))):
                    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
                    deg_ang = int(np.rad2deg(angle))
                    if deg_ang >= 0:
                        az[ind] = 90+deg_ang
                    else:
                        az[ind] = 270 + deg_ang
                    ind =ind+1
                    ax.plot(origin, (y0, y1), '-r')
                ax.set_xlim(origin)
                ax.set_ylim((edges.shape[0], 0))
                ax.set_axis_off()
                unique_elements, counts_elements = np.unique(az, return_counts=True)
            
                check = counts_elements[np.argmax(counts_elements)]
                
                if check == 1:
                    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1, threshold =0.25*np.max(h))):
                        deg_ang = int(np.rad2deg(angle))
                        if deg_ang >= 0:
                            azimuth = 90+deg_ang
                        else:
                            azimuth = 270 + deg_ang
                else:
                    azimuth = (unique_elements[np.argmax(counts_elements)])
                    #print(np.asarray((unique_elements, counts_elements)))
                    ax.set_title('Azimuth = %i' %azimuth)
                #save the image
                if save_img_file_path != None:
                    plt.savefig(save_img_file_path + '/crop_mask_az_'+str(ii),
                                dpi=300)
                #Show the plot if plot_show = True
                if plot_show == True:
                    plt.tight_layout()
                    plt.show()     
            elif no_figs == 3:
                ax = ax.ravel()

                ax[0].imshow(in_img, cmap=cm.gray)
                ax[0].set_title('Input image')
                ax[0].set_axis_off()
    

                ax[1].imshow(np.log(1 + h),
                    extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                    cmap=cm.gray, aspect=1/1.5)
                ax[1].set_title('Hough transform')
                ax[1].set_xlabel('Angles (degrees)')
                ax[1].set_ylabel('Distance (pixels)')
                ax[1].axis('image')

                ax[2].imshow(in_img)# cmap=cm.gray)
                origin = np.array((0, edges.shape[1]))
                ind =0
                for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=no_lines, threshold =0.25*np.max(h))):
                    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
                
                    deg_ang = int(np.rad2deg(angle))
                    if deg_ang >= 0:
                        az[ind] = 90+deg_ang
                    else:
                        az[ind] = 270 + deg_ang
                    ind =ind+1
                    ax.plot(origin, (y0, y1), '-r')
                ax[2].set_xlim(origin)
                ax[2].set_ylim((edges.shape[0], 0))
                ax[2].set_axis_off()
                unique_elements, counts_elements = np.unique(az, return_counts=True)
            
                check = counts_elements[np.argmax(counts_elements)]
                
                if check == 1:
                    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1, threshold =0.25*np.max(h))):
                        deg_ang = int(np.rad2deg(angle))
                        if deg_ang >= 0:
                            azimuth = 90+deg_ang
                        else:
                            azimuth = 270 + deg_ang
                else:
                    azimuth = (unique_elements[np.argmax(counts_elements)])
                    #print(np.asarray((unique_elements, counts_elements)))
                    ax[2].set_title('Azimuth = %i' %azimuth)
                #save the image
                if save_img_file_path != None:
                    plt.savefig(save_img_file_path + '/crop_mask_az_'+str(ii),
                                dpi=300)
                #Show the plot if plot_show = True
                if plot_show == True:
                    plt.tight_layout()
                    plt.show() 
            else:
                print("Enter valid parameters")
    