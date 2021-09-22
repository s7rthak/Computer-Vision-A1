import cv2
import updated_flow
import numpy as np

BASELINE_FRAMES = 1700
input_path = "COL780-A1-Data/illumination/input/"

of = updated_flow.OpticalFlow()
bg = updated_flow.ModelBackground()

for i in range(1, BASELINE_FRAMES+1):
   frame2 = cv2.imread(input_path + "in" + str(i).zfill(6) + ".jpg")   
   of.insert_image(frame2)
   flow_magnitude = of.compute_optical_flow()
   mask_of = of.get_mask_of_moving()
   bg.insert_image(frame2)
        # bg_img = bg.get_background_image()
   mask_bg = bg.get_mask_of_foreground(frame2)
   mask2 = (mask_bg * mask_of)**(0.9)
   img_disp_combine = np.hstack( (frame2, updated_flow.mask2image( mask2 ),updated_flow.setMaskOntoImage(mask2, frame2) ))

   img_disp =np.hstack( (frame2, updated_flow.mask2image( mask_of )))
   cv2.imshow("optical flow", img_disp)
        # Waitkey
   q = cv2.waitKey(1)
   if q!=-1 and chr(q) == 'q':
       break
  
cv2.destroyAllWindows()